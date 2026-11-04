"""The native extractor: three model passes per chunk, deterministic validation, coverage and cost accounting.

Per chunk: detect (candidates by line reference) -> enrich (assertions with people, modality and times) ->
deterministic checks (`validate.check_item`) -> verify (entailment, modality, attribution verdicts). Results are
deduplicated across chunks and capped at the request limits. Nothing is persisted here; B6's pipeline persists
the `ExtractionResult` and resumes from its coverage map.

Model calls go through `memory_learning.extraction_request` (OpenAI Responses, strict JSON schema), which holds a
budget reservation per call and records usage under the current cost feature; this module sets that feature to
`context_extraction` and enforces the per-source cap. A call whose upper-bound cost would exceed what remains of
the cap is not made: its chunk and every later chunk are `skipped` with reason `source_budget` (or
`owner_budget` when the monthly budget refuses the reservation) and the result is marked partial. A chunk whose
call fails or returns unusable output is `failed`; later chunks still run.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import httpx
from pydantic import ValidationError

from ... import memory_learning, model_map
from ...agent_models import catalog
from ...cost_features import current_feature
from ...domain import DomainError
from .. import settings as context_settings
from ..schema import Coverage, ExtractionResult, ExtractRequest, ProposedAction
from .chunking import TextChunk, chunk_segments
from .prompts import PASSES, PROMPT_VERSION, meeting_block, payload
from .validate import Draft, Rejection, SourceContext, apply_verdicts, candidate_key, check_item, dedupe

log = logging.getLogger(__name__)

FEATURE = "context_extraction"  # cost_features.LABELS entry added by package A2
JOB = "context_extraction"  # model_map job; until it is registered, the extraction model is used directly
BUDGET_CODES = {"BUDGET_EXCEEDED", "BUDGET_DEFERRED"}


class BudgetStop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ChunkFailed(Exception):
    pass


def limits() -> tuple[float, float]:
    """(model timeout seconds, per-source budget USD)."""
    settings = context_settings.current()
    return float(settings.model_timeout_s), float(settings.source_budget_usd)


def call_model(owner: str, body: dict, allowance: float, timeout: float, prompt_tokens: int) -> dict:
    if JOB in model_map.JOBS:
        return memory_learning.profile_request(owner, JOB, body, allowance, timeout, prompt_tokens=prompt_tokens)
    return memory_learning.extraction_request(owner, "chat/completions", body, memory_learning.EXTRACTION_MODEL,
                                              allowance, timeout)


@dataclass
class Spend:
    cap: float
    spent: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ChunkRun:
    chunk: TextChunk
    status: str = "ok"  # ok | failed | skipped
    reason: str = ""
    calls: int = 0
    cost: float = 0.0
    drafts: list[Draft] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


class NativeExtractor:
    """Implements `engine.Extractor`."""

    def __init__(self, *, target_tokens: int = 5000, neighbour_tokens: int = 500, timeout_s: float | None = None,
                 budget_usd: float | None = None):
        self.target_tokens, self.neighbour_tokens = target_tokens, neighbour_tokens
        self.timeout_s, self.budget_usd = timeout_s, budget_usd

    def extract(self, request: ExtractRequest) -> ExtractionResult:
        timeout, cap = limits()
        timeout = self.timeout_s or timeout
        spend = Spend(cap=cap if self.budget_usd is None else self.budget_usd)
        ctx = SourceContext.from_request(request)
        meeting = {
            "title": request.title, "meeting_local_date": ctx.local_date.isoformat(),
            "timezone": request.timezone or "UTC",
            "participants": [p.model_dump(exclude_none=True) for p in request.participants],
        }
        candidates = [{"id": candidate_key(c.ref), "type": c.ref.entity_type, "name": c.name,
                       "aliases": c.aliases, "detail": c.detail[:300]} for c in request.candidates]
        runs = [ChunkRun(c) for c in chunk_segments(request.segments, self.target_tokens, self.neighbour_tokens)]
        token = current_feature.set(FEATURE)
        try:
            stop = None
            for run in runs:
                if stop:
                    run.status, run.reason = "skipped", stop
                    continue
                try:
                    self._chunk(request.owner_id, run, ctx, meeting, candidates, spend, timeout)
                except BudgetStop as budget:
                    stop = budget.reason
                    run.status, run.reason = "skipped", stop
                    run.drafts.clear()
                except ChunkFailed as error:
                    run.status, run.reason = "failed", str(error)[:300]
                    run.drafts.clear()
        finally:
            current_feature.reset(token)
        return assemble(request, runs, spend)

    # ------------------------------------------------------------ one chunk

    def _chunk(self, owner, run, ctx, meeting, candidates, spend, timeout) -> None:
        block = meeting_block(meeting, run.chunk)
        detection = self._call(owner, "context_detect", block, spend, run, timeout)
        if not detection.candidates:
            return
        enrichment = self._call(owner, "context_enrich", {
            **block, "candidates": [c.model_dump() for c in detection.candidates],
            "entity_candidates": candidates}, spend, run, timeout)
        drafts = []
        for item in enrichment.items:
            outcome = check_item(item, ctx, run.chunk.owns)
            (drafts if isinstance(outcome, Draft) else run.rejected).append(outcome)
        if not drafts:
            return
        review = [{"item": i, "kind": d.kind, "statement": d.statement,
                   "modality": d.assertion.modality if d.assertion else "proposed",
                   "responsible": [r.mention or r.entity_id for r in d.responsible],
                   "due": d.due.expression if d.due else None,
                   "evidence": [{"segment_seq": e.segment_seq, "speaker": ctx.segments[e.segment_seq].speaker_label,
                                 "quote": e.quote} for e in d.evidence]}
                  for i, d in enumerate(drafts)]
        verification = self._call(owner, "context_verify", {**block, "items": review}, spend, run, timeout)
        kept, rejected = apply_verdicts(drafts, verification.verdicts)
        run.drafts, run.rejected = kept, run.rejected + rejected

    def _call(self, owner: str, name: str, user: dict, spend: Spend, run: ChunkRun, timeout: float) -> Any:
        system, model, max_tokens = PASSES[name]
        body = payload(system, user, name, model, max_tokens)
        prompt_tokens = model_map.estimate_tokens(system) + model_map.estimate_tokens(body["messages"][1]["content"])
        agent = catalog()["luna"]
        bound = replace(agent, max_output_tokens=max(8192, max_tokens)).reserve_cost(prompt_tokens)
        if spend.spent + bound > spend.cap:
            raise BudgetStop("source_budget")
        spend.calls += 1
        run.calls += 1
        try:
            data = call_model(owner, body, bound, timeout, prompt_tokens)
        except DomainError as error:
            if error.code in BUDGET_CODES:
                raise BudgetStop("owner_budget") from error
            if error.code != "INTEGRATION_UNAVAILABLE":  # raised before any request is sent
                self._charge(spend, run, bound)
            raise ChunkFailed(f"{name}: {error.code}") from error
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            self._charge(spend, run, bound)  # outcome unknown: count the upper bound against the cap
            raise ChunkFailed(f"{name}: {type(error).__name__}") from error
        usage = agent.usage_record(data)
        spend.prompt_tokens += usage["prompt_tokens"]
        spend.completion_tokens += usage["completion_tokens"]
        self._charge(spend, run, agent.usage_cost(usage))
        message = data["choices"][0]["message"]
        if message.get("refusal"):
            raise ChunkFailed(f"{name}: refusal")
        if data["choices"][0].get("finish_reason") == "length":
            raise ChunkFailed(f"{name}: truncated")
        try:
            return model.model_validate_json(message.get("content") or "")
        except ValidationError as error:
            raise ChunkFailed(f"{name}: invalid output") from error

    @staticmethod
    def _charge(spend: Spend, run: ChunkRun, amount: float) -> None:
        spend.spent += amount
        run.cost += amount


# ---------------------------------------------------------------- assembly

PRIORITY = {"failed": 3, "skipped": 2, "extracted": 1, "no_content": 0}


def assemble(request: ExtractRequest, runs: list[ChunkRun], spend: Spend) -> ExtractionResult:
    drafts, rejected = dedupe(d for run in runs for d in run.drafts)
    rejected = [r for run in runs for r in run.rejected] + rejected
    assertions, actions, links, derivations, notes, cited = [], [], {}, {}, {}, set()
    for draft in drafts:
        if draft.assertion is not None:
            if len(assertions) >= request.limits.max_assertions:
                rejected.append(draft.reject("limit", "max_assertions"))
                continue
            index = len(assertions)
            assertions.append(draft.assertion)
            if draft.derivations:
                derivations[str(index)] = draft.derivations
            if draft.notes:
                notes[str(index)] = draft.notes
            for link in draft.links:
                links.setdefault((link.mention.casefold(), link.candidate.entity_type, link.candidate.entity_id),
                                 link)
        if draft.kind in ("commitment", "suggestion"):
            if len(actions) >= request.limits.max_actions:
                rejected.append(draft.reject("limit", "max_actions"))
            else:
                actions.append(ProposedAction(
                    kind="create" if draft.kind == "commitment" else "suggestion", title=draft.statement[:500],
                    responsible=draft.responsible, due=draft.due, evidence=draft.evidence,
                    assertion_indexes=[len(assertions) - 1] if draft.kind == "commitment" else []))
        cited.update(draft.seqs)

    coverage: dict[int, Coverage] = {}
    reasons: dict[str, dict[str, str]] = {"skipped": {}, "failed": {}}
    for run in runs:
        for seq in run.chunk.segment_seqs:
            status = run.status if run.status != "ok" else "extracted" if seq in cited else "no_content"
            if PRIORITY[status] >= PRIORITY.get(coverage.get(seq, "no_content"), 0):
                coverage[seq] = status
            if run.status != "ok":
                reasons[run.status][str(seq)] = run.reason
    uncovered = sorted(seq for seq, status in coverage.items() if status in ("skipped", "failed"))
    diagnostics = {
        "engine": "native", "prompt_version": PROMPT_VERSION, "model": memory_learning.EXTRACTION_MODEL,
        "feature": FEATURE, "partial": bool(uncovered), "uncovered_segments": uncovered, **reasons,
        "chunks": [{"seq": r.chunk.seq, "segment_start": r.chunk.spans[0].seq, "segment_end": r.chunk.spans[-1].seq,
                    "token_count": r.chunk.token_count, "status": r.status, "reason": r.reason, "calls": r.calls,
                    "cost_usd": round(r.cost, 6)} for r in runs],
        "cost_usd": round(spend.spent, 6), "budget_usd": spend.cap,
        "usage": {"calls": spend.calls, "prompt_tokens": spend.prompt_tokens,
                  "completion_tokens": spend.completion_tokens},
        "rejected": [r.as_dict() for r in rejected],
        "rejection_counts": dict(sorted(Counter(r.code for r in rejected).items())),
        "derivations": derivations, "notes": notes,
    }
    return ExtractionResult(assertions=assertions, links=list(links.values()), actions=actions, coverage=coverage,
                            diagnostics=json.loads(json.dumps(diagnostics, default=str)))

