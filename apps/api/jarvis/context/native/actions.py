"""The action-item pass: every follow-up anyone will do after a meeting, from one call over the whole meeting.

Input: all transcript lines (`[12] Speaker: text`), the owner's private notes as `[N1] text` lines flagged as typed
by the recorder during the meeting, the participants with the recorder named, and the meeting's local date.
Meetings above `WHOLE_MEETING_TOKENS` fall back to `CHUNK_TOKENS` chunks (private notes in every call) whose items
are merged deterministically. The model returns items with an owner and an `owner_basis` (prompts.ActionItem).

Deterministic checks per item (`check`):
- evidence: each quote must occur verbatim in the line it cites (`validate.ground_evidence`); quotes that do not
  are dropped and listed in the item's signals, and an item left with none is rejected (`quote_not_found`);
- owner: the owner is the speaker of a cited line (a private note's speaker is the recorder), is named in the
  quotes or the private notes, or, for implied items, is someone in the meeting (`attribution_mismatch`
  otherwise). On lines without a speaker, a participant the model names that the lines can't prove is kept as an
  inferred owner (`owner_basis` "inferred", never checked) unless the quotes show someone else took it; a
  non-participant, "Unknown speaker" and items cited only from shorthand notes stay unassigned (an explicit
  to-do note is the recorder's);
- numbers: every number and identifier in the title occurs somewhere in the meeting (lines and private notes);
  an identifier like `C2` is one token, never the number 2, and "1-to-5" is 1 and 5 (`ungrounded_number`). A
  description sentence with an ungrounded number is dropped (signal `description_ungrounded`), never the item;
- due: resolved as in the chunked passes; an expression not in the quotes or a wrong derivation drops the due
  date (recorded), never the item;
- modality: hypothetical cues or explicit declines in the owner's quotes, with no clean quote beside them, reject
  (`hypothetical`, `negated`), as does an owner's decline or withdrawal ("scrap that") with no later commitment;
  a condition on planned work ("if approval arrives") is not a hypothetical; promise and tentative cues are signals that adjust `owner_basis` (a "promise"
  without promise language becomes intent or implied), not rejections;
- work finished in the meeting (`done_in_meeting`) and prior Wrong/Dismiss decisions (`prior_decision`) reject.

Then one verify call ("a new follow-up or a progress report? is it real, and is the owner right?") rejects
progress reports (`progress_report`), no (`not_a_follow_up`) or a wrong owner (`owner_rejected`; an inferred
owner it contradicts is blanked instead). Kept items become `ProposedAction`s; `stated_promise` and `accepted_request`
items also become commitment assertions so open loops and briefs keep working.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ... import model_map
from .. import settings as context_settings
from ..identity import OWNER_ACTOR_NAME
from ..schema import (
    CandidateAssertion,
    EntityRef,
    EvidenceRef,
    ExtractRequest,
    LinkProposal,
    Segment,
    TimeBound,
)
from .chunking import chunk_segments
from .prompts import PROMPT_VERSION, ActionItem, ActionList, ActionVerdict, ActionVerification
from .validate import (
    COLLECTIVE,
    COLLECTIVE_CUE,
    HYPOTHETICAL,
    IDENTIFIER,
    PROMISE,
    QUOTE_CHARS,
    TENS,
    TENTATIVE,
    UNITS,
    UNKNOWN,
    SourceContext,
    _word_in,
    ground_evidence,
    links_for,
    normalized,
    resolve_time,
    ungrounded_tokens,
)

JOB = "context_actions"
WHOLE_MEETING_TOKENS = 60_000
CHUNK_TOKENS, CONTEXT_TOKENS = 20_000, 1_000
MAX_EVIDENCE = 8
VERIFY_NEIGHBOURS = 2
MERGE = 0.75  # title-word Jaccard (without STOPWORDS) for the same owner's duplicate across chunks
STOPWORDS = {"the", "and", "for", "with", "from", "into", "onto", "that", "this", "then", "their", "our", "your"}
COMMITTING = ("stated_promise", "accepted_request")
INFERRED = "inferred"  # owner_basis for a likely owner the lines can't prove (unseparated speakers)

NEGATED = re.compile(
    r"\b(won't|will not|not going to|no need to|don't need to|do not need to|let's not|lets not|"
    r"we're not|we are not|i'm not|i am not|decided against|decided not to|not doing|skip that|drop that|"
    r"i can't|i cannot|i can not|not able to|unable to|no time to)\b",
    re.IGNORECASE)
TODO_NOTE = re.compile(r"(?:^|\n)\s*(?:[-*\u2022]\s*)?(?:\[\s?\]\s*)?(?:todo|to-?do|action item|my action)s?\b|"
                       r"\b(?:i need to|i'll|i will|i must|i have to|i've got to|remind me to|note to self)\b",
                       re.IGNORECASE)
WITHDRAWN = re.compile(r"\b(scrap that|scratch that|never ?mind|forget (?:it|that|about it)|cancel that|"
                       r"on second thought|ignore that)\b", re.IGNORECASE)
# Taking work (back) on: firmer than PROMISE, which also counts a bare "yes", "sure" or "okay".
RECOMMIT = re.compile(r"\b(i'll|i will|i can(?!'t|not| not)|i'm going to|i am going to|let me|happy to|will do|"
                      r"on it|i've got (?:it|this)|leave it with me|i'll take|i'll handle)\b", re.IGNORECASE)
NAMED_PROMISE = r"\b{}\b\s*(?:'ll|will|is going to|can|agreed to|to)\b"


def route(prompt_tokens: int = 0) -> model_map.Route:
    """The job's route: an in-process MODEL_MAP override (the meeting lab) wins, then the `actions_model`
    setting, then the model map's default."""
    configured = context_settings.current().actions_model
    if configured and model_map.MODEL_MAP.get(JOB) == model_map.DEFAULTS[JOB]:
        return model_map.route(JOB, prompt_tokens, model_map={**model_map.MODEL_MAP, JOB: configured})
    return model_map.route(JOB, prompt_tokens)


# ---------------------------------------------------------------- the meeting as the model sees it


def _line(segment: Segment) -> str:
    return f"[{segment.seq}] {segment.speaker_label or 'Unknown speaker'}: {segment.text}"


@dataclass
class Meeting:
    request: ExtractRequest
    ctx: SourceContext
    transcript: list[Segment]
    notes: dict[str, Segment]  # "N1" -> private note segment
    recorder: str | None
    actors: dict[str, set[str]]  # lowercased name or alias -> actor IDs among the candidates

    @classmethod
    def of(cls, request: ExtractRequest, ctx: SourceContext) -> "Meeting":
        ordered = sorted(request.segments, key=lambda s: s.seq)
        transcript = [s for s in ordered if s.kind == "transcript"]
        notes = {f"N{i}": s for i, s in enumerate((s for s in ordered if s.kind == "private_note"), 1)}
        recorder = next((s.speaker_label for s in transcript if s.attribution == "owner" and s.speaker_label), None)
        recorder = recorder or next((p.speaker_label or p.name for p in request.participants
                                     if p.role == "organizer" and (p.speaker_label or p.name)), None)
        actors: dict[str, set[str]] = {}
        for c in request.candidates:
            if c.ref.entity_type != "actor":
                continue
            names = [n for n in (c.name, *c.aliases) if n and n != OWNER_ACTOR_NAME]
            if c.name == OWNER_ACTOR_NAME and not c.aliases and recorder:
                names.append(recorder)  # the user's own meeting: the recorder is the user
            for name in names:
                actors.setdefault(name.casefold().strip(), set()).add(c.ref.entity_id)
        return cls(request, ctx, transcript, notes, recorder, actors)

    def ref(self, seq: int) -> str:
        return next((k for k, s in self.notes.items() if s.seq == seq), str(seq))

    def seq(self, ref: str) -> int | None:
        ref = ref.strip().strip("[]").upper()
        if ref in self.notes:
            return self.notes[ref].seq
        return int(ref) if ref.isdigit() and int(ref) in self.ctx.segments else None

    def speaker(self, seq: int) -> str | None:
        segment = self.ctx.segments[seq]
        return self.recorder if segment.kind == "private_note" else segment.speaker_label

    def people(self) -> list[dict[str, Any]]:
        people = [{**p.model_dump(exclude_none=True, exclude={"role"}), "role": p.role,
                   "recorded_the_meeting": bool(self.recorder and self.recorder in (p.speaker_label, p.name))}
                  for p in self.request.participants]
        known = {p.get("speaker_label") or p.get("name") for p in people}
        people += [{"speaker_label": label, "recorded_the_meeting": label == self.recorder}
                   for label in dict.fromkeys(s.speaker_label for s in self.transcript if s.speaker_label)
                   if label not in known]
        return people

    def payload(self, lines: str, before: str = "", after: str = "") -> dict[str, Any]:
        notes = "\n".join(f"[{k}] {s.text}" for k, s in self.notes.items()) or "(none)"
        out = {"title": self.request.title, "meeting_local_date": self.ctx.local_date.isoformat(),
               "timezone": self.request.timezone or "UTC", "recorded_by": self.recorder or "unknown",
               "people": self.people(), "lines": lines or "(none)",
               "private_notes": f"Typed by {self.recorder or 'the recorder'} during the meeting:\n{notes}"
               if self.notes else "(none)"}
        if before or after:
            out.update(context_before=before or "(none)", context_after=after or "(none)")
        return out

    def calls(self) -> tuple[str, list[tuple[dict[str, Any], Callable[[int, int, int], bool]]]]:
        """("whole" | "chunked", [(user payload, owns)])."""
        whole = "\n".join(_line(s) for s in self.transcript)
        notes = {s.seq for s in self.notes.values()}
        if model_map.estimate_tokens(whole) <= WHOLE_MEETING_TOKENS:
            return "whole", [(self.payload(whole), lambda seq, a, b: True)]
        out = []
        for chunk in chunk_segments(self.transcript, CHUNK_TOKENS, CONTEXT_TOKENS):
            out.append((self.payload(chunk.text(), _spans(chunk.before), _spans(chunk.after)),
                        lambda seq, a, b, chunk=chunk: seq in notes or chunk.owns(seq, a, b)))
        return "chunked", out


def _spans(spans: tuple) -> str:
    return "\n".join(s.line() for s in spans)


# ---------------------------------------------------------------- one item


@dataclass
class ActionDraft:
    kind: str  # action | suggestion
    title: str
    description: str
    owner: str | None
    basis: str
    model_basis: str
    recipient: str | None
    evidence: list[EvidenceRef]
    responsible: list[EntityRef] = field(default_factory=list)
    due: TimeBound | None = None
    confidence: float = 0.0
    derivations: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    assertion: CandidateAssertion | None = None
    links: list[LinkProposal] = field(default_factory=list)
    verified: bool | None = None
    verdict: str = ""
    action_index: int | None = None
    assertion_index: int | None = None
    status: str = "open"  # open | ongoing (a status update the checker judges)

    @property
    def seqs(self) -> list[int]:
        return list(dict.fromkeys(e.segment_seq for e in self.evidence))

    def describe(self, meeting: Meeting) -> dict[str, Any]:
        return {"action_index": self.action_index, "assertion_index": self.assertion_index, "kind": self.kind,
                "status": self.status, "title": self.title, "description": self.description, "owner": self.owner,
                "owner_basis": self.basis, "model_owner_basis": self.model_basis, "recipient": self.recipient,
                "responsible": [r.model_dump(exclude_none=True) for r in self.responsible],
                "due": self.due.model_dump(mode="json") if self.due else None, "confidence": self.confidence,
                "verified": self.verified, "verdict": self.verdict, "signals": self.signals,
                "derivations": self.derivations,
                "evidence": [{"segment": e.segment_seq, "line": meeting.ref(e.segment_seq),
                              "speaker": meeting.speaker(e.segment_seq), "quote": e.quote} for e in self.evidence]}


def rejection(code: str, reason: str, title: str, segments) -> dict[str, Any]:
    return {"code": code, "reason": reason[:300], "title": title[:500], "segments": list(segments)}


def _names(name: str, meeting: Meeting) -> set[str]:
    """Lowercased names a person goes by: the name, a speaker label's participant names, actor aliases."""
    key = name.casefold().strip()
    names = {key} | meeting.ctx.speaker_names.get(name, set())
    for label, known in meeting.ctx.speaker_names.items():
        if key in known or key == label.casefold():
            names |= known | {label.casefold()}
    ids = meeting.actors.get(key, set())
    names |= {n for n, owners in meeting.actors.items() if owners & ids}
    return {n for n in names if n}


def _first(names: set[str]) -> set[str]:
    return {n.split()[0] for n in names if n.split()}


def resolve(name: str | None, meeting: Meeting) -> EntityRef | None:
    """An actor reference for a person the model named: the candidate actor when exactly one matches the name
    (or, failing that, its first name), else an unresolved mention."""
    if not name or not name.strip():
        return None
    mention = name.strip()[:300]
    key = mention.casefold()
    if key in COLLECTIVE or key in UNKNOWN:
        return EntityRef(entity_type="actor", mention=mention)
    names = _names(mention, meeting)
    ids = set().union(*(meeting.actors.get(n, set()) for n in names))
    if not ids:
        firsts = _first(names)
        ids = set().union(*(owners for n, owners in meeting.actors.items() if n.split() and n.split()[0] in firsts))
    if len(ids) == 1:
        return EntityRef(entity_type="actor", entity_id=next(iter(ids)), mention=mention)
    return EntityRef(entity_type="actor", mention=mention)


def _owner_evidence(owner: str | None, evidence: list[EvidenceRef], meeting: Meeting) -> list[EvidenceRef]:
    """The cited lines the owner said (a private note is the recorder's)."""
    if not owner:
        return []
    key = owner.casefold().strip()
    if key in UNKNOWN:  # the unattributed lines
        return [e for e in evidence if meeting.speaker(e.segment_seq) is None]
    if key in COLLECTIVE:
        return [e for e in evidence if COLLECTIVE_CUE.search(e.quote)]
    names = _names(owner, meeting)
    firsts = _first(names)
    out = []
    for e in evidence:
        speaker = (meeting.speaker(e.segment_seq) or "").casefold()
        spoken = _names(speaker, meeting) if speaker else set()
        if spoken & names or _first(spoken) & firsts:
            out.append(e)
    return out


def _owner_quotes(owner: str | None, evidence: list[EvidenceRef], meeting: Meeting) -> list[str]:
    return [e.quote.translate(QUOTE_CHARS) for e in _owner_evidence(owner, evidence, meeting)]


def _participant(name: str, meeting: Meeting) -> str | None:
    """The meeting participant (name or speaker label) `name` refers to, if any."""
    for person in meeting.people():
        for label in (person.get("name"), person.get("speaker_label")):
            if label and _same_person(name, label, meeting):
                return label
    return None


def _someone_else_took_it(owner: str, evidence: list[EvidenceRef], meeting: Meeting) -> str | None:
    """Another participant the quotes show taking the work ("this is Aarish, I'll take that", "Mustafa will")."""
    quotes = " ".join(e.quote for e in evidence).translate(QUOTE_CHARS)
    for person in meeting.people():
        name = person.get("name") or person.get("speaker_label")
        if not name or _same_person(name, owner, meeting):
            continue
        for n in _names(name, meeting) | _first(_names(name, meeting)):
            if not n:
                continue
            claimed = re.search(NAMED_PROMISE.format(re.escape(n)), quotes, re.IGNORECASE) or re.search(
                rf"\b(?:this is|i'm|i am|it's)\s+{re.escape(n)}\b", quotes, re.IGNORECASE)
            if claimed:
                return name
    return None


def _same_person(a: str, b: str, meeting: Meeting) -> bool:
    x, y = _names(a, meeting), _names(b, meeting)
    return bool(x & y or _first(x) & _first(y))


def _taken_back(owned: list[EvidenceRef]) -> bool:
    """The owner's last word on it declines or withdraws ("No, I can't this week", "actually, scrap that") with no
    later commitment; a bare "yes" about something else does not take it back up."""
    said = sorted((e.segment_seq, e.quote.translate(QUOTE_CHARS)) for e in owned)
    declined = [seq for seq, q in said if (NEGATED.search(q) or WITHDRAWN.search(q)) and not RECOMMIT.search(q)]
    committed = [seq for seq, q in said if RECOMMIT.search(q)]
    return bool(declined) and (not committed or max(declined) > max(committed))


def _unattributed(evidence: list[EvidenceRef], meeting: Meeting) -> list[str]:
    """Quotes from transcript lines without a speaker (unseparated audio)."""
    return [e.quote.translate(QUOTE_CHARS) for e in evidence
            if meeting.ctx.segments[e.segment_seq].kind == "transcript" and meeting.speaker(e.segment_seq) is None]


def _attributed(owner: str, basis: str, evidence: list[EvidenceRef], meeting: Meeting, title: str = "") -> bool:
    key = owner.casefold().strip()
    quotes = " ".join(e.quote for e in evidence).translate(QUOTE_CHARS).casefold()
    if key in UNKNOWN:
        return any(meeting.ctx.segments[e.segment_seq].kind == "transcript" and meeting.speaker(e.segment_seq) is None
                   for e in evidence)
    if key in COLLECTIVE:
        return bool(COLLECTIVE_CUE.search(quotes))
    if _owner_quotes(owner, evidence, meeting):
        return True
    names = _names(owner, meeting)
    if any(_word_in(n, quotes) for n in names | _first(names)):
        return True
    for note in meeting.notes.values():  # a recorder's note about this work that names the owner
        text = note.text.translate(QUOTE_CHARS).casefold()
        if any(_word_in(n, text) for n in names | _first(names)) and _words(title) & _words(note.text):
            return True
    if basis == "implied_by_discussion":  # implied work falls to someone in the meeting
        present = {n for p in meeting.people() for n in (p.get("name"), p.get("speaker_label")) if n}
        return any(_names(p, meeting) & names or _first(_names(p, meeting)) & _first(names) for p in present)
    return False


def _basis(item: ActionItem, owner_quotes: list[str], evidence: list[EvidenceRef], meeting: Meeting
           ) -> tuple[str, dict[str, Any]]:
    """The owner basis after the promise and tentative signals, and the signals themselves."""
    names = _names(item.owner, meeting) if item.owner else set()
    named = [re.compile(NAMED_PROMISE.format(re.escape(n)), re.IGNORECASE) for n in names | _first(names) if n]
    promising = [q for q in owner_quotes if PROMISE.search(q)]
    promising += [e.quote for e in evidence if any(p.search(e.quote.translate(QUOTE_CHARS)) for p in named)]
    signals: dict[str, Any] = {}
    unattributed = _unattributed(evidence, meeting)
    quotes = " ".join(e.quote for e in evidence).translate(QUOTE_CHARS).casefold()
    if unattributed and not owner_quotes and any(_word_in(n, quotes) for n in names | _first(names)):
        # Speakers not separated: "Davis, can you..." then "Sure, I'll do it" on an unattributed line.
        promising += [q for q in unattributed if PROMISE.search(q)]
        signals["owner_from_content"] = True
    tentative = bool(promising) and all(TENTATIVE.search(q) for q in promising)
    signals.update(promise=bool(promising), tentative=tentative, owner_spoke=bool(owner_quotes))
    basis = item.owner_basis
    if basis in COMMITTING and not promising:
        basis = "self_assigned_intent" if owner_quotes else "implied_by_discussion"
    elif basis == "stated_promise" and tentative:
        basis = "self_assigned_intent"
    return basis, signals


def check(item: ActionItem, meeting: Meeting, owns: Callable[[int, int, int], bool]) -> ActionDraft | dict:
    title = " ".join(item.title.split())[:500]
    cited = [(meeting.seq(q.line), q.quote) for q in item.evidence[:MAX_EVIDENCE]]
    evidence, dropped = [], []
    for seq, quote in cited:
        refs, code, _ = ground_evidence([(seq, quote)], meeting.ctx, owns) if seq is not None else ([], "unknown", "")
        if code:
            dropped.append(quote[:200])
        elif refs[0] not in evidence:
            evidence += refs
    known = [seq for seq, _ in cited if seq is not None]
    if not title:
        return rejection("invalid", "empty title", item.title, known)
    if not evidence:
        return rejection("quote_not_found", "; ".join(dropped), title, known)
    seqs = list(dict.fromkeys(e.segment_seq for e in evidence))
    if item.status == "done_in_meeting":
        return rejection("done_in_meeting", "finished during the meeting", title, seqs)
    owner = (item.owner or "").strip() or None
    unassigned: dict[str, Any] = {}
    inferred = taker = False
    if item.kind == "suggestion":
        owner = None
    elif owner and owner.casefold() in UNKNOWN:
        owner, unassigned = None, {"owner_unknown": True}  # promised on a line without a speaker: unclear owner
    elif (owner and _unattributed(evidence, meeting)
          and not _attributed(owner, item.owner_basis, evidence, meeting, title)):
        # Unseparated speakers: the lines don't show who spoke and nothing names the model's owner. A participant
        # the model names stays as an inferred owner (never checked), unless the quotes show someone else took
        # it; a name outside the meeting leaves the item unassigned. Never dropped.
        if _participant(owner, meeting) and not (taker := _someone_else_took_it(owner, evidence, meeting)):
            inferred = True
            unassigned = {"owner_inferred": owner[:300]}
        else:
            reason = f"taken by {taker}" if _participant(owner, meeting) else "not in the meeting"
            owner, unassigned = None, {"owner_unverified": owner[:300], "owner_unverified_reason": reason}
    if owner and not any(meeting.ctx.segments[e.segment_seq].kind == "transcript" for e in evidence):
        # Only the recorder's shorthand notes: they assign nothing ("Cynthia annotation admin") unless a note is
        # an explicit personal to-do ("todo: send Yvette the deck", "I need to..."), which is the recorder's.
        notes = " ".join(meeting.ctx.segments[seq].text for seq in seqs)
        if TODO_NOTE.search(notes) and meeting.recorder:
            if not _same_person(owner, meeting.recorder, meeting):
                unassigned = {"owner_from_note": owner[:300]}
            owner = meeting.recorder
        else:
            owner, unassigned = None, {"note_only": owner[:300]}
    owned = _owner_evidence(owner, evidence, meeting)
    owner_quotes = [e.quote.translate(QUOTE_CHARS) for e in owned]
    primary = owner_quotes or [e.quote.translate(QUOTE_CHARS) for e in evidence]
    if any(HYPOTHETICAL.search(q) for q in primary) and not any(not HYPOTHETICAL.search(q) for q in primary):
        return rejection("hypothetical", "hypothetical cue in every quote", title, seqs)
    if any(NEGATED.search(q) for q in primary) and not any(not NEGATED.search(q) for q in primary):
        return rejection("negated", "declined or ruled out in every quote", title, seqs)
    if item.owner_basis != "implied_by_discussion" and _taken_back(owned):
        return rejection("negated", "the owner declined or withdrew it after", title, seqs)
    if owner and not inferred and not _attributed(owner, item.owner_basis, evidence, meeting, title):
        return rejection("attribution_mismatch", owner, title, seqs)

    quotes = " ".join(e.quote for e in evidence)
    derivations, due = [], None
    bound, record, code = resolve_time("due", item.due, quotes, meeting.ctx)
    if record:
        derivations.append({**record, **({"dropped": code} if code else {})})
    if not code:
        due = bound
    derived = [due.at.astimezone(meeting.ctx.zone)] if due else []
    everything = " ".join(s.text for s in meeting.request.segments)  # transcript and private notes
    if missing := ungrounded_tokens(title, everything, derived):
        return rejection("ungrounded_number", "title: " + ", ".join(missing), title, seqs)
    # The description never rejects the item: sentences with numbers found nowhere in the meeting are dropped.
    description, unfounded = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(item.description.split())[:5000]):
        if missing := ungrounded_tokens(sentence, everything, derived):
            unfounded += missing
        elif sentence:
            description.append(sentence)
    if normalized(title) in meeting.ctx.blocked:
        return rejection("prior_decision", "dismissed or marked wrong before", title, seqs)

    basis, signals = _basis(item, owner_quotes, evidence, meeting) if owner else ("implied_by_discussion", {})
    if inferred:
        basis = INFERRED  # a likely owner the lines can't prove: never a commitment, never checked
    signals.update(unassigned)
    if unfounded:
        signals["description_ungrounded"] = list(dict.fromkeys(unfounded))
    if dropped:
        signals["dropped_quotes"] = dropped
    responsible = [r for r in (resolve(owner, meeting),) if r]
    draft = ActionDraft(kind=item.kind, status=item.status, title=title, description=" ".join(description),
                        owner=owner, basis=basis,
                        model_basis=item.owner_basis, recipient=(item.recipient or "").strip()[:300] or None,
                        evidence=evidence, responsible=responsible, due=due,
                        confidence=max(0.0, min(1.0, item.confidence)), derivations=derivations, signals=signals)
    if item.kind == "action" and basis in COMMITTING and responsible:
        draft.assertion = CandidateAssertion(
            kind="commitment", statement=commitment_statement(owner, title, due)[:2000], responsible=responsible,
            recipient=resolve(draft.recipient, meeting), due=due, evidence=evidence, modality="asserted",
            confidence=draft.confidence)
        draft.links = links_for(draft.assertion, evidence)
    return draft


def commitment_statement(owner: str, title: str, due: TimeBound | None) -> str:
    """"Priya Natarajan will send the data export by Friday." from an imperative title and the due expression."""
    verb = title.rstrip(" .")
    if len(verb) > 1 and not verb[1].isupper():  # keep acronyms ("QA the build")
        verb = verb[0].lower() + verb[1:]
    expression = (due.expression or "").strip() if due else ""
    if expression and expression.lower() not in verb.lower():
        verb = f"{verb} {expression}"
    return f"{owner} will {verb}."


# ---------------------------------------------------------------- merge and verify


def _words(title: str) -> set[str]:
    """Title words for near-matching: words over two letters, plus identifiers (C2, v2: anything with a digit)
    and short uppercase words (QA, UI), which tell otherwise equal titles apart."""
    tokens = re.findall(r"[A-Za-z0-9]+", title.translate(QUOTE_CHARS))
    words = {t.casefold() for t in tokens if (len(t) > 2 and t.casefold() not in STOPWORDS)
             or any(c.isdigit() for c in t) or (len(t) == 2 and t.isupper())}
    return {str(UNITS.get(w, TENS.get(w, w))) for w in words}  # "fifty" and "50" are one word


def _distinct(a: ActionDraft, b: ActionDraft) -> bool:
    """Near-equal titles that are still separate deliverables: different identifiers (C2 vs C3), different
    recipients, or two different due dates."""
    ids = [{m.casefold() for m in IDENTIFIER.findall(d.title)} for d in (a, b)]
    recipients = [(d.recipient or "").casefold().strip() for d in (a, b)]
    return (ids[0] != ids[1] or (all(recipients) and recipients[0] != recipients[1])
            or (a.due is not None and b.due is not None and a.due.at != b.due.at))


def merge(drafts: list[ActionDraft]) -> tuple[list[ActionDraft], list[dict]]:
    """One item per deliverable: the same owner's items with near-equal titles fold their evidence together,
    unless `_distinct` tells them apart."""
    def who(d: ActionDraft) -> str:
        return (d.responsible[0].entity_id if d.responsible else None) or (d.owner or "").casefold()

    def same(a: ActionDraft, b: ActionDraft) -> bool:
        x, y = _words(a.title), _words(b.title)
        near = bool(x and y) and len(x & y) / len(x | y) >= MERGE
        # An unassigned copy of an owned item (a chunk that saw only the note) joins it when they share a line.
        owners = who(a) == who(b) or ((not a.owner or not b.owner) and bool(set(a.seqs) & set(b.seqs)))
        return (owners and a.kind == b.kind and (normalized(a.title) == normalized(b.title) or near)
                and not _distinct(a, b))

    kept: list[ActionDraft] = []
    rejected = []
    for draft in drafts:
        twin = next((k for k in kept if same(k, draft)), None)
        if twin is None:
            kept.append(draft)
            continue
        if twin.owner is None and draft.owner:  # keep the owned one, fold the unassigned copy into it
            kept[kept.index(twin)], twin, draft = draft, draft, twin
        seen = {(e.segment_seq, e.char_start) for e in twin.evidence}
        twin.evidence += [e for e in draft.evidence if (e.segment_seq, e.char_start) not in seen][
            :max(0, MAX_EVIDENCE - len(twin.evidence))]
        twin.due = twin.due or draft.due
        twin.description = twin.description or draft.description
        rejected.append(rejection("duplicate", f"merged into {twin.title[:200]}", draft.title, draft.seqs))
    return kept, rejected


def review(drafts: list[ActionDraft], meeting: Meeting) -> dict[str, Any]:
    """The verify call's input: the items with speakers, and the lines around their evidence."""
    cited = {seq for d in drafts for seq in d.seqs}
    near = sorted(cited | {s.seq for s in meeting.transcript
                           if any(abs(s.seq - seq) <= VERIFY_NEIGHBOURS for seq in cited)})
    context = [f"[{meeting.ref(seq)}] Private note: {meeting.ctx.segments[seq].text}"
               if meeting.ctx.segments[seq].kind == "private_note" else _line(meeting.ctx.segments[seq])
               for seq in near]
    return {"title": meeting.request.title, "meeting_local_date": meeting.ctx.local_date.isoformat(),
            "recorded_by": meeting.recorder or "unknown", "people": meeting.people(), "context": "\n".join(context),
            "items": [{"item": i, "title": d.title, "description": d.description, "owner": d.owner,
                       "owner_basis": d.basis, "kind": d.kind, "status": d.status,
                       "evidence": [{"line": meeting.ref(e.segment_seq), "speaker": meeting.speaker(e.segment_seq),
                                     "quote": e.quote} for e in d.evidence]}
                      for i, d in enumerate(drafts)]}


def apply_verdicts(drafts: list[ActionDraft], verdicts: list[ActionVerdict]) -> tuple[list[ActionDraft], list[dict]]:
    by_item = {v.item: v for v in verdicts}
    kept, rejected = [], []
    for i, draft in enumerate(drafts):
        verdict = by_item.get(i)
        if verdict is None:  # the verifier skipped it: kept, marked unverified
            kept.append(draft)
            continue
        draft.verified, draft.verdict = (verdict.real_follow_up and verdict.owner_correct
                                         and not verdict.progress_report), verdict.reason[:300]
        if verdict.progress_report:
            rejected.append(rejection("progress_report", verdict.reason or "a status update, not a follow-up",
                                      draft.title, draft.seqs))
        elif not verdict.real_follow_up:
            rejected.append(rejection("not_a_follow_up", verdict.reason, draft.title, draft.seqs))
        elif not verdict.owner_correct and draft.owner and draft.basis == INFERRED:
            # An inferred owner the checker contradicts: the work stays, unassigned.
            draft.signals["owner_rejected"] = draft.owner
            draft.owner, draft.responsible, draft.basis, draft.verified = None, [], "implied_by_discussion", True
            kept.append(draft)
        elif not verdict.owner_correct and draft.owner:
            rejected.append(rejection("owner_rejected", verdict.reason, draft.title, draft.seqs))
        else:
            kept.append(draft)
    return kept, rejected


# ---------------------------------------------------------------- the pass


@dataclass
class ActionRun:
    """The pass's outcome and accounting (`calls`, `cost` and tokens are charged by the extractor's `_call`)."""

    status: str = "ok"  # ok | failed | skipped
    reason: str = ""
    mode: str = "whole"
    profile: str | None = None
    model: str | None = None
    calls: int = 0
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    drafts: list[ActionDraft] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    def diagnostics(self, meeting: Meeting) -> dict[str, Any]:
        return {"model": self.model, "profile": self.profile, "prompt_version": PROMPT_VERSION,
                "status": self.status, "reason": self.reason, "mode": self.mode, "recorded_by": meeting.recorder,
                "items": [d.describe(meeting) for d in self.drafts], "rejected": self.rejected,
                "calls": self.calls, "cost_usd": round(self.cost, 6), "latency_ms": self.latency_ms,
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}}


    # Persistence for callers that resume runs (the pipeline saves completed chunks the same way).

    def dump(self, key: str) -> dict[str, Any]:
        """A JSON-safe record of a completed pass, restorable by `restore` when `cache_key` still matches."""
        def model(value):
            return None if value is None else value.model_dump(mode="json")
        drafts = [{**{k: getattr(d, k) for k in DRAFT_FIELDS}, "evidence": [model(e) for e in d.evidence],
                   "responsible": [model(r) for r in d.responsible], "due": model(d.due),
                   "assertion": model(d.assertion), "links": [model(x) for x in d.links]} for d in self.drafts]
        return json.loads(json.dumps({
            "key": key, "mode": self.mode, "profile": self.profile, "model": self.model, "calls": self.calls,
            "cost": self.cost, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms, "drafts": drafts, "rejected": self.rejected}, default=str))

    def restore(self, data: dict[str, Any], meeting: Meeting) -> None:
        """Load a `dump` (status stays ok; `cost` is what the saved calls cost, for the caller to charge). Items
        dismissed or marked wrong since it was saved are rejected again, as a fresh pass would."""
        for name in ("mode", "profile", "model", "calls", "cost", "prompt_tokens", "completion_tokens",
                     "latency_ms"):
            setattr(self, name, data[name])
        self.rejected = list(data["rejected"])
        self.drafts = [ActionDraft(
            **{k: d[k] for k in DRAFT_FIELDS}, evidence=[EvidenceRef.model_validate(e) for e in d["evidence"]],
            responsible=[EntityRef.model_validate(r) for r in d["responsible"]],
            due=TimeBound.model_validate(d["due"]) if d["due"] else None,
            assertion=CandidateAssertion.model_validate(d["assertion"]) if d["assertion"] else None,
            links=[LinkProposal.model_validate(x) for x in d["links"]]) for d in data["drafts"]]
        blocked = [d for d in self.drafts if normalized(d.title) in meeting.ctx.blocked]
        self.drafts = [d for d in self.drafts if d not in blocked]
        self.rejected += [rejection("prior_decision", "dismissed or marked wrong before", d.title, d.seqs)
                          for d in blocked]


DRAFT_FIELDS = ("kind", "status", "title", "description", "owner", "basis", "model_basis", "recipient", "confidence",
                "derivations", "signals", "verified", "verdict")


def cache_key(meeting: Meeting) -> str:
    """What the pass's calls depend on: the prompt version, the routed profile and the request (revision, lines,
    notes, participants, candidates, timezone). Decisions are left out, as for saved chunks: `restore` rechecks
    them. A reprocess never sees an earlier run's saved pass (the pipeline reads only its own runs')."""
    request = meeting.request.model_dump(mode="json", exclude={"decisions", "limits"})
    raw = json.dumps({"prompt": PROMPT_VERSION, "profile": route().profile, "request": request}, sort_keys=True,
                     default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


Call = Callable[[str, dict[str, Any]], Any]  # (pass name, user payload) -> validated model output


def run(meeting: Meeting, call: Call, out: ActionRun) -> None:
    """Fill `out` with checked, merged and verified drafts. Exceptions from `call` (budget stops, failed calls)
    propagate; `out` keeps the accounting of the calls made so far."""
    started = time.monotonic()
    try:
        out.mode, calls = meeting.calls()
        drafts = []
        for user, owns in calls:
            listed: ActionList = call("context_actions", user)
            for item in listed.items:
                outcome = check(item, meeting, owns)
                (drafts if isinstance(outcome, ActionDraft) else out.rejected).append(outcome)
        drafts, duplicates = merge(drafts)
        out.rejected += duplicates
        if drafts:
            verification: ActionVerification = call("context_actions_verify", review(drafts, meeting))
            drafts, refused = apply_verdicts(drafts, verification.verdicts)
            out.rejected += refused
        out.drafts = drafts
    finally:
        out.latency_ms = int((time.monotonic() - started) * 1000)
