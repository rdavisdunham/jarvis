"""Deterministic implementations of the four engine protocols, for tests and offline development.

- `DoubleExtractor` returns a canned `ExtractionResult` keyed by `ExtractRequest.content_hash` when it has one,
  and otherwise applies rules: a sentence starting "Decision:" is a decision; "Action:", "I'll" or "I will"
  is a commitment (with a proposed task). Quotes are exact; ISO dates (2026-11-20) in a commitment become its
  due date; candidate names and aliases found in the text become inferred link proposals.
- `DoubleReconciler` merges identical statements, and supersedes keyed values (`EXCLUSIVE_KEYS`) per
  `(subject, key)` in `valid_from` order; equal `valid_from` with different statements is a conflict.
- `DoubleRetriever` scores an in-memory corpus lexically after applying the scope predicate.
- `DoubleSynthesizer` groups assertions into the six brief sections and lists gaps.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, time
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

from .schema import (
    AssertionView,
    BriefClaim,
    BriefSection,
    CandidateAssertion,
    EntityRef,
    EvidenceRef,
    ExtractionResult,
    ExtractRequest,
    Hit,
    LinkProposal,
    ProposedAction,
    ReconcileOp,
    ReconcileRequest,
    RetrieveQuery,
    ScopePredicate,
    SynthesizeRequest,
    TimeBound,
)

SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
ACTION_OWNER = re.compile(r"^Action:\s*([A-Z][\w'-]*(?: [A-Z][\w'-]*)?) (?:to|will)\b")
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
WORD = re.compile(r"\w+")


# ---------------------------------------------------------------- extract


class DoubleExtractor:
    def __init__(self, canned: Mapping[str, ExtractionResult] | None = None):
        self.canned = dict(canned or {})

    @classmethod
    def from_directory(cls, path: str | Path) -> "DoubleExtractor":
        """Each `<content_hash>.json` file holds one serialized ExtractionResult."""
        return cls({f.stem: ExtractionResult.model_validate_json(f.read_text())
                    for f in sorted(Path(path).glob("*.json"))})

    def extract(self, request: ExtractRequest) -> ExtractionResult:
        canned = self.canned.get(request.content_hash)
        return canned if canned is not None else rule_based(request)


def _due(statement: str, timezone: str | None) -> TimeBound | None:
    match = ISO_DATE.search(statement)
    if not match:
        return None
    day = datetime.combine(datetime.strptime(match.group(0), "%Y-%m-%d").date(), time(), ZoneInfo(timezone or "UTC"))
    return TimeBound(at=day, precision="day", expression=match.group(0))


def _interpret(sentence: str, speaker: str | None) -> tuple[str, str, str | None] | None:
    """(kind, statement, responsible mention) for a sentence the rules recognise."""
    if sentence.startswith("Decision:"):
        return "decision", sentence.removeprefix("Decision:").strip(), None
    if sentence.startswith("Action:"):
        owner = ACTION_OWNER.match(sentence)
        return "commitment", sentence.removeprefix("Action:").strip(), owner.group(1) if owner else speaker
    if sentence.startswith(("I'll ", "I will ")):
        return "commitment", sentence, speaker
    return None


def rule_based(request: ExtractRequest) -> ExtractionResult:
    assertions, actions, links, coverage = [], [], [], {}
    for segment in request.segments:
        found = False
        for match in SENTENCE.finditer(segment.text):
            sentence = match.group(0).strip()
            parsed = _interpret(sentence, segment.speaker_label) if sentence else None
            if parsed is None or len(assertions) >= request.limits.max_assertions:
                continue
            kind, statement, owner = parsed
            if not statement:
                continue
            start = segment.text.index(sentence, match.start())
            evidence = [EvidenceRef(segment_seq=segment.seq, quote=sentence, char_start=start,
                                    char_end=start + len(sentence))]
            commitment = kind == "commitment"
            responsible = [EntityRef(entity_type="actor", mention=owner or "unknown speaker")] if commitment else []
            due = _due(statement, request.timezone) if commitment else None
            assertions.append(CandidateAssertion(kind=kind, statement=statement, responsible=responsible, due=due,
                                                 key="due_date" if due else None, evidence=evidence))
            if commitment and len(actions) < request.limits.max_actions:
                actions.append(ProposedAction(kind="create", title=statement[:500], responsible=responsible, due=due,
                                              assertion_indexes=[len(assertions) - 1], evidence=evidence))
            found = True
        for candidate in request.candidates:
            for name in (candidate.name, *candidate.aliases):
                hit = re.search(rf"\b{re.escape(name)}\b", segment.text)
                if hit:
                    links.append(LinkProposal(
                        mention=name, candidate=candidate.ref, status="inferred", score=1.0,
                        evidence=[EvidenceRef(segment_seq=segment.seq, quote=name, char_start=hit.start(),
                                              char_end=hit.end())]))
                    break
        coverage[segment.seq] = "extracted" if found else "no_content"
    return ExtractionResult(assertions=assertions, links=links, actions=actions, coverage=coverage,
                            diagnostics={"engine": "double", "mode": "rules"})


# ---------------------------------------------------------------- reconcile


def _normalized(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def _order(a: AssertionView):
    return a.valid_from.at, a.recorded_at, a.id


class DoubleReconciler:
    def reconcile(self, request: ReconcileRequest) -> list[ReconcileOp]:
        pool = {a.id: a for a in [*request.related, *request.changed] if a.lifecycle in ("current", "contested")}
        changed = {a.id for a in request.changed if a.id in pool}
        ops, touched = [], set()

        def emit(op, pair, explanation):
            ops.append(ReconcileOp(op=op, targets=[a.id for a in pair], explanation=explanation,
                                   target_revisions={a.id: a.revision for a in pair}))
            touched.update(a.id for a in pair)

        keyed = defaultdict(list)
        for a in sorted(pool.values(), key=_order):
            if a.subject and a.subject.entity_id and a.key in request.exclusive_keys:
                keyed[(a.subject.entity_type, a.subject.entity_id, a.key)].append(a)
        for (_, _, key), rows in sorted(keyed.items()):
            for old, new in pairwise(rows):
                if not ({old.id, new.id} & changed):
                    continue
                if _normalized(old.statement) == _normalized(new.statement):
                    emit("equivalent", (old, new), f"same {key}")
                elif old.valid_from.at == new.valid_from.at:
                    emit("conflict", (old, new), f"different {key} effective at the same time")
                else:
                    emit("supersede", (old, new), f"later {key} replaces the earlier one")

        seen = {}
        for a in sorted(pool.values(), key=_order):
            sig = (a.kind, a.subject, _normalized(a.statement))
            first = seen.setdefault(sig, a)
            if first is not a and {first.id, a.id} & changed and not {first.id, a.id} <= touched:
                emit("equivalent", (first, a), "identical statement")
        for a in sorted(request.changed, key=_order):
            if a.id not in touched:
                ops.append(ReconcileOp(op="no_change", targets=[a.id], target_revisions={a.id: a.revision}))
        return ops


# ---------------------------------------------------------------- retrieve


@dataclass(frozen=True)
class Indexed:
    """A retrievable item: the hit to return plus what scoping and scoring need."""

    owner_id: str
    hit: Hit
    text: str
    recorded_from: datetime | None = None
    recorded_to: datetime | None = None


def in_scope(item: Indexed, scope: ScopePredicate, include_raw: bool) -> bool:
    hit = item.hit
    if item.owner_id != scope.owner_id:
        return False
    if hit.ref_type != "assertion" and not include_raw:
        return False
    if hit.lifecycle == "shadow" and hit.ref_type == "assertion":
        return False
    if (hit.lifecycle == "superseded" and not scope.include_superseded) or (
            hit.lifecycle == "retracted" and not scope.include_retracted):
        return False
    if scope.kinds and hit.ref_type == "assertion" and hit.kind not in scope.kinds:
        return False
    if scope.entities:
        wanted = {(e.entity_type, e.entity_id) for e in scope.entities}
        if not wanted & {(e.entity_type, e.entity_id) for e in hit.entities}:
            return False
    return _within(hit.valid_from, hit.valid_to, scope.valid_at) and _within(
        item.recorded_from, item.recorded_to, scope.known_at)


def _within(start: datetime | None, end: datetime | None, at: datetime | None) -> bool:
    """`at` falls in the half-open `[start, end)`; open bounds and a missing `at` always pass."""
    return at is None or ((start is None or start <= at) and (end is None or at < end))


class DoubleRetriever:
    def __init__(self, corpus: Iterable[Indexed] = ()):
        self.corpus = list(corpus)

    def retrieve(self, query: RetrieveQuery) -> list[Hit]:
        terms = set(WORD.findall(query.text.lower()))
        scored = []
        for item in self.corpus:
            if not in_scope(item, query.scope, query.include_raw):
                continue
            words = WORD.findall(item.text.lower())
            matched = terms & set(words)
            if matched:
                score = len(matched) / len(terms) + sum(words.count(t) for t in matched) / (1000 * len(words))
                scored.append((round(score, 6), item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].hit.ref_type, pair[1].hit.id))
        return [item.hit.model_copy(update={"score": score, "rank": {"lexical_rank": float(rank)},
                                            "snippet": item.hit.snippet or item.text[:200]})
                for rank, (score, item) in enumerate(scored[:query.limit], start=1)]


# ---------------------------------------------------------------- synthesize

SECTION_KINDS = {
    "current_state_and_why": ("fact", "decision", "rationale", "preference", "procedure"),
    "open_loops": ("commitment",),
    "next_meeting_prep": ("open_question", "risk"),
    "who_is_who": ("relationship",),
    "unknowns": ("open_question",),
}


def _claim(a: AssertionView) -> BriefClaim:
    text = f"{a.statement} (contested)" if a.lifecycle == "contested" else a.statement
    return BriefClaim(text=text, assertion_ids=[a.id])


class DoubleSynthesizer:
    def synthesize(self, request: SynthesizeRequest) -> list[BriefSection]:
        current = sorted((a for a in request.assertions if a.lifecycle in ("current", "contested")), key=_order)
        tasks_of = defaultdict(list)
        for task in sorted(request.tasks, key=lambda t: t.id):
            for assertion_id in task.assertion_ids:
                tasks_of[assertion_id].append(task.id)
        current_ids = {a.id for a in current}
        sections = []
        for name in request.brief.sections:
            gaps = []
            if name == "recent_changes":
                claims = [BriefClaim(text=f"No longer current ({a.lifecycle}): {a.statement}", assertion_ids=[a.id])
                          for a in sorted(request.history, key=_order)]
            else:
                claims = [_claim(a) for a in current if a.kind in SECTION_KINDS[name]]
            if name == "open_loops":
                claims = [c.model_copy(update={"task_ids": tasks_of[c.assertion_ids[0]]}) for c in claims]
                claims += [BriefClaim(text=f"Task: {t.title} ({t.status})", task_ids=[t.id])
                           for t in sorted(request.tasks, key=lambda t: t.id)
                           if not current_ids & set(t.assertion_ids)]
                gaps += [f"No due date for: {a.statement}" for a in current if a.kind == "commitment" and not a.due]
            if not claims:
                gaps.append(f"Nothing recorded for {name.replace('_', ' ')}.")
            sections.append(BriefSection(name=name, claims=claims, gaps=gaps))
        return sections


def canned_file(directory: str | Path, request: ExtractRequest, result: ExtractionResult) -> Path:
    """Write `result` as the canned output for `request` (a helper for recording fixtures)."""
    path = Path(directory) / f"{request.content_hash}.json"
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return path
