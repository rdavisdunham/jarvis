"""The native extractor: a whole-meeting action pass, three model passes per chunk, deterministic validation,
coverage and cost accounting.

First the action pass (actions.py) lists every follow-up from the whole meeting: its items become
`ProposedAction`s, and promised or accepted ones also commitment assertions. Then per chunk: detect (candidates by
line reference) -> enrich (assertions with people, modality and times) -> deterministic checks
(`validate.check_item`) -> verify (entailment, modality, attribution verdicts). The chunk passes no longer produce
commitments or suggestions. Results are deduplicated across chunks and capped at the request limits. Nothing is
persisted here; B6's pipeline persists the `ExtractionResult` and resumes from its coverage map.

Model calls go through `memory_learning.profile_request`, routed by job (`context_extraction` for the chunk passes,
`context_actions` for the action pass; `model_map.MODEL_MAP` or the `actions_model` setting choose the profile)
and priced by `model_map.costing(profile)`. Each call holds a budget reservation and records usage under the
`context_extraction` cost feature; this module enforces the per-source cap. A call whose upper-bound cost would
exceed what remains of the cap is not made: its chunk and every later chunk are `skipped` with reason
`source_budget` (or `owner_budget` when the monthly budget refuses the reservation) and the result is marked
partial. A chunk whose call fails or returns unusable output is `failed`; later chunks still run. A failed or
skipped action pass marks every segment failed or skipped, so the source stays partial and is retried, and no
proposal is superseded on an incomplete list.

Extraction modes (`extraction_mode()`): "split" (the default) is the flow above. "combined" replaces the action pass
and the per-chunk passes with one whole-meeting call (`context_combined`, the action pass's chunk fallback above
`actions.WHOLE_MEETING_TOKENS`) that returns both the assertions (the enrich fields, cited by line id) and the
action items, then one verify call over both (`context_combined_verify`). Every deterministic step is the split
mode's: `validate.check_item` and `validate.apply_verdicts` for assertions, `actions.check`, `merge` and
`apply_verdicts` for action items, then the same dedupe, limits and coverage. Both calls route and price as the
`context_actions` job and are accounted on the action run (`diagnostics["actions"]`);
`diagnostics["extraction_mode"]` and `diagnostics["combined"]` record the mode, calls and cost. The pipeline
caches the split mode's action pass and chunks; a combined run is not cached, so a retry calls the model again.

Selecting the mode: the setting `context_extraction_mode` (env `JARVIS_CONTEXT_EXTRACTION_MODE`, "split" or
"combined"), or in-process for the meeting lab, `extract.EXTRACTION_MODE = "combined"` (None defers to the setting),
set before the extract job runs and restored afterwards.
"""

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import httpx
from pydantic import ValidationError

from ... import memory_learning, model_map
from ...cost_features import current_feature
from ...domain import DomainError
from .. import settings as context_settings
from ..schema import Coverage, ExtractionResult, ExtractRequest, ProposedAction
from . import actions as action_pass
from .chunking import TextChunk, chunk_segments
from .prompts import (
    ACTION_PASSES,
    COMBINED_PASSES,
    COMBINED_PROMPT_VERSION,
    PASSES,
    PROMPT_VERSION,
    CombinedAssertion,
    CombinedList,
    CombinedVerification,
    Enriched,
    Quote,
    meeting_block,
    payload,
)
from .validate import Draft, Rejection, SourceContext, apply_verdicts, candidate_key, check_item, dedupe

log = logging.getLogger(__name__)

FEATURE = "context_extraction"  # cost_features.LABELS entry added by package A2
JOB = "context_extraction"  # model_map job of the chunk passes
BUDGET_CODES = {"BUDGET_EXCEEDED", "BUDGET_DEFERRED"}
ACTION_KINDS = ("commitment", "suggestion")  # the action pass owns these
MODES = ("split", "combined")
EXTRACTION_MODE: str | None = None  # in-process override (the meeting lab); None defers to the setting
MAX_QUOTES, QUOTE_CHARS, STATEMENT_CHARS = 6, 1000, 2000  # Enriched's limits, applied to combined assertions


def extraction_mode() -> str:
    mode = EXTRACTION_MODE or context_settings.current().extraction_mode
    if mode not in MODES:
        raise ValueError(f"Unknown extraction mode {mode!r}; expected one of {MODES}")
    return mode


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


def call_model(owner: str, body: dict, allowance: float, timeout: float, prompt_tokens: int, *, job: str = JOB,
               profile: str | None = None) -> dict:
    return memory_learning.profile_request(owner, job, body, allowance, timeout, prompt_tokens=prompt_tokens,
                                           profile=profile)


def route(job: str, prompt_tokens: int) -> model_map.Route:
    return action_pass.route(prompt_tokens) if job == action_pass.JOB else model_map.route(job, prompt_tokens)


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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    drafts: list[Draft] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


class NativeExtractor:
    """Implements `engine.Extractor`."""

    def __init__(self, *, target_tokens: int = 5000, neighbour_tokens: int = 500, timeout_s: float | None = None,
                 budget_usd: float | None = None):
        self.target_tokens, self.neighbour_tokens = target_tokens, neighbour_tokens
        self.timeout_s, self.budget_usd = timeout_s, budget_usd

    def extract(self, request: ExtractRequest) -> ExtractionResult:
        mode = extraction_mode()
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
        acts, found = action_pass.ActionRun(fallback=action_pass.fallback()), action_pass.Meeting.of(request, ctx)
        token = current_feature.set(FEATURE)
        try:
            if mode == "combined":
                self._run_combined(request.owner_id, found, acts, runs, candidates, spend, timeout)
                return assemble(request, runs, spend, acts, found, mode=mode)
            stop = None
            try:
                self._actions(request.owner_id, found, acts, spend, timeout)
            except BudgetStop as budget:
                stop = budget.reason
                acts.status, acts.reason, acts.drafts = "skipped", stop, []
            except ChunkFailed as error:
                acts.status, acts.reason, acts.drafts = "failed", str(error)[:300], []
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
        return assemble(request, runs, spend, acts, found, mode=mode)

    # ------------------------------------------------------------ combined mode

    def _run_combined(self, owner, meeting: action_pass.Meeting, acts: action_pass.ActionRun, runs: list[ChunkRun],
                      candidates: list[dict], spend: Spend, timeout: float) -> None:
        """Combined mode in place of the action pass and the chunk loop. Kept assertions go on the first chunk run
        (assembly dedupes across runs and covers segments by citation); a stopped or failed call marks the action
        run and every chunk run, as a failed action pass does in split mode."""
        try:
            drafts, rejected = self._combined(owner, meeting, acts, candidates, spend, timeout)
        except (BudgetStop, ChunkFailed) as error:
            status = "skipped" if isinstance(error, BudgetStop) else "failed"
            reason = error.reason if isinstance(error, BudgetStop) else str(error)[:300]
            acts.status, acts.reason, acts.drafts = status, reason, []
            for run in runs:
                run.status, run.reason = status, reason
            return
        if runs:
            runs[0].drafts, runs[0].rejected = drafts, rejected

    def _combined(self, owner, meeting: action_pass.Meeting, out: action_pass.ActionRun, candidates: list[dict],
                  spend: Spend, timeout: float) -> tuple[list[Draft], list[Rejection]]:
        """One call for both lists (chunked as the action pass is), the split mode's checks, one verify call.
        Action items land in `out`; returns the kept assertions and the rejected ones."""
        slow = _slow(timeout)
        started = time.monotonic()
        notes: list[Draft] = []
        refused: list[Rejection] = []
        try:
            out.mode, calls = meeting.calls()
            items = []
            for user, owns in calls:
                listed: CombinedList = self._call(
                    owner, "context_combined", {**user, "entity_candidates": candidates}, spend, out, slow,
                    passes=COMBINED_PASSES, job=action_pass.JOB)
                for item in listed.actions:
                    outcome = action_pass.check(item, meeting, owns)
                    (items if isinstance(outcome, action_pass.ActionDraft) else out.rejected).append(outcome)
                for index, assertion in enumerate(listed.assertions):
                    outcome = enriched(assertion, index, meeting, owns)
                    (notes if isinstance(outcome, Draft) else refused).append(outcome)
            items, duplicates = action_pass.merge(items)
            out.rejected += duplicates
            if items or notes:
                verification: CombinedVerification = self._call(
                    owner, "context_combined_verify", combined_review(items, notes, meeting), spend, out, slow,
                    passes=COMBINED_PASSES, job=action_pass.JOB)
                items, rejected = action_pass.apply_verdicts(items, verification.action_verdicts)
                out.rejected += rejected
                notes, unverified = apply_verdicts(notes, verification.assertion_verdicts)
                refused += unverified
            out.drafts = items
        finally:
            out.latency_ms = int((time.monotonic() - started) * 1000)
        return notes, refused

    # ------------------------------------------------------------ the action pass

    def _actions(self, owner, meeting: action_pass.Meeting, out: action_pass.ActionRun, spend, timeout) -> None:
        """The whole-meeting action pass. A seam for callers that cache it (the pipeline caches chunks only)."""
        slow = _slow(timeout)
        action_pass.run(meeting, lambda name, user: self._call(
            owner, name, user, spend, out, slow if name == "context_actions" else timeout,
            passes=ACTION_PASSES, job=action_pass.JOB), out)

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
            if item.kind in ACTION_KINDS:
                continue  # never asked for per chunk: the action pass lists follow-ups
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

    def _call(self, owner: str, name: str, user: dict, spend: Spend, run, timeout: float, *,
              passes: dict = PASSES, job: str = JOB) -> Any:
        """One priced model call charged to `spend` and `run` (a ChunkRun or an ActionRun)."""
        system, model, max_tokens = passes[name]
        body = payload(system, user, name, model, max_tokens)
        prompt_tokens = model_map.estimate_tokens(system) + model_map.estimate_tokens(body["messages"][1]["content"])
        profile = route(job, prompt_tokens).profile
        agent = model_map.costing(profile)
        if agent is None:
            raise ChunkFailed(f"{name}: model profile {profile} has no price")
        if hasattr(run, "profile"):
            run.profile, run.model = profile, agent.model
        bound = replace(agent, max_output_tokens=max(8192, max_tokens)).reserve_cost(prompt_tokens)
        if spend.spent + bound > spend.cap:
            raise BudgetStop("source_budget")
        spend.calls += 1
        run.calls += 1
        try:
            data = call_model(owner, body, bound, timeout, prompt_tokens, job=job, profile=profile)
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
        run.prompt_tokens += usage["prompt_tokens"]
        run.completion_tokens += usage["completion_tokens"]
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
    def _charge(spend: Spend, run, amount: float) -> None:
        spend.spent += amount
        run.cost += amount


def _slow(timeout: float) -> float:
    """One call reads the whole meeting and writes the whole list; high reasoning takes longer still."""
    effort = model_map.PROFILES[action_pass.route().profile].reasoning_effort
    return max(2 * timeout, 600.0 if effort in ("high", "xhigh", "max") else 180.0)


def enriched(item: CombinedAssertion, index: int, meeting: action_pass.Meeting, owns) -> Draft | Rejection:
    """A combined-mode assertion as the enrich pass's item (line ids to segment numbers, clipped to Enriched's
    limits), then the split mode's `check_item`."""
    statement = item.statement.strip()[:STATEMENT_CHARS]
    quotes = [(meeting.seq(q.line), q.quote[:QUOTE_CHARS]) for q in item.evidence if q.quote.strip()]
    unknown = [q.line for q in item.evidence if meeting.seq(q.line) is None]
    evidence = [Quote(segment_seq=seq, quote=quote) for seq, quote in quotes if seq is not None][:MAX_QUOTES]
    if not evidence:
        detail = f"line {', '.join(unknown)[:200]}" if unknown else "no quote"
        return Rejection("unknown_segment" if unknown else "quote_not_found", statement, detail)
    try:
        candidate = Enriched(
            candidate=index, kind=item.kind, statement=statement, subject=item.subject, object=item.object,
            key=item.key, modality=item.modality, inferred=item.inferred, responsible=item.responsible,
            responsibility=item.responsibility, recipient=item.recipient, due=None, valid_from=item.valid_from,
            valid_to=item.valid_to, evidence=evidence, confidence=min(max(item.confidence, 0.0), 1.0))
    except ValidationError as error:
        return Rejection("invalid", statement, str(error.errors()[0].get("msg", ""))[:200],
                         tuple(q.segment_seq for q in evidence))
    return check_item(candidate, meeting.ctx, owns)


def combined_review(items: list, notes: list[Draft], meeting: action_pass.Meeting) -> dict:
    """The combined verify call's input: the action pass's review of the items beside the chunk verify pass's
    review of the assertions (cited by line id), with the lines around all their evidence."""
    out = action_pass.review(items, meeting)
    cited = {seq for d in items for seq in d.seqs} | {seq for d in notes for seq in d.seqs}
    out["context"] = action_pass.context(cited, meeting)
    out["actions"] = out.pop("items")
    out["assertions"] = [{"item": i, "kind": d.kind, "statement": d.statement,
                          "modality": d.assertion.modality if d.assertion else "proposed",
                          "responsible": [r.mention or r.entity_id for r in d.responsible],
                          "evidence": [{"line": meeting.ref(e.segment_seq), "speaker": meeting.speaker(e.segment_seq),
                                        "quote": e.quote} for e in d.evidence]}
                         for i, d in enumerate(notes)]
    return out


# ---------------------------------------------------------------- assembly

PRIORITY = {"failed": 3, "skipped": 2, "extracted": 1, "no_content": 0}


def assemble(request: ExtractRequest, runs: list[ChunkRun], spend: Spend, acts: action_pass.ActionRun | None = None,
             meeting: action_pass.Meeting | None = None, *, mode: str = "split") -> ExtractionResult:
    acts = acts or action_pass.ActionRun(status="skipped", reason="not run")
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
    _add_actions(request, acts, assertions, actions, links, derivations, rejected, cited)

    coverage: dict[int, Coverage] = {}
    reasons: dict[str, dict[str, str]] = {"skipped": {}, "failed": {}}
    if acts.status != "ok":  # an incomplete follow-up list leaves the whole source to retry
        for segment in request.segments:
            coverage[segment.seq] = acts.status
            reasons[acts.status][str(segment.seq)] = f"actions: {acts.reason}"
    for run in runs:
        for seq in run.chunk.segment_seqs:
            status = run.status if run.status != "ok" else "extracted" if seq in cited else "no_content"
            if PRIORITY[status] >= PRIORITY.get(coverage.get(seq, "no_content"), 0):
                coverage[seq] = status
            if run.status != "ok":
                reasons[run.status][str(seq)] = run.reason
    uncovered = sorted(seq for seq, status in coverage.items() if status in ("skipped", "failed"))
    profile = model_map.PROFILES.get(model_map.route(JOB).profile if mode == "split" else action_pass.route().profile)
    diagnostics = {
        "engine": "native", "prompt_version": PROMPT_VERSION,
        "model": profile.model if profile else memory_learning.EXTRACTION_MODEL,
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
        "actions": acts.diagnostics(meeting or action_pass.Meeting.of(request, SourceContext.from_request(request))),
        "extraction_mode": mode,
    }
    if mode == "combined":
        diagnostics["combined"] = {
            "prompt_version": COMBINED_PROMPT_VERSION, "profile": acts.profile, "model": acts.model,
            "status": acts.status, "calls": acts.calls, "cost_usd": round(acts.cost, 6),
            "latency_ms": acts.latency_ms,
            "assertions": len(assertions) - sum(1 for d in acts.drafts if d.assertion_index is not None),
            "action_items": len(acts.drafts)}
    return ExtractionResult(assertions=assertions, links=list(links.values()), actions=actions, coverage=coverage,
                            diagnostics=json.loads(json.dumps(diagnostics, default=str)))



def _add_actions(request: ExtractRequest, acts: action_pass.ActionRun, assertions: list, actions: list, links: dict,
                 derivations: dict, rejected: list, cited: set) -> None:
    """The action pass's items as proposed actions, with commitment assertions for promised or accepted work."""
    for draft in acts.drafts:
        index = None
        if draft.assertion is not None:
            if len(assertions) >= request.limits.max_assertions:
                rejected.append(Rejection("limit", draft.title, "max_assertions", tuple(draft.seqs)))
            else:
                index = len(assertions)
                assertions.append(draft.assertion)
                if draft.derivations:
                    derivations[str(index)] = draft.derivations
                for link in draft.links:
                    links.setdefault((link.mention.casefold(), link.candidate.entity_type,
                                      link.candidate.entity_id), link)
        draft.assertion_index = index
        if len(actions) >= request.limits.max_actions:
            rejected.append(Rejection("limit", draft.title, "max_actions", tuple(draft.seqs)))
            continue
        draft.action_index = len(actions)
        actions.append(ProposedAction(
            kind="create" if draft.kind == "action" else "suggestion", title=draft.title,
            description=draft.description, responsible=draft.responsible, due=draft.due, evidence=draft.evidence,
            assertion_indexes=[] if index is None else [index]))
        cited.update(draft.seqs)
    rejected += [Rejection(r["code"], r["title"], r["reason"], tuple(r["segments"])) for r in acts.rejected]
