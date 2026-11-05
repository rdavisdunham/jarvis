"""Deterministic checks on enriched extraction items, and the application of pass-3 verdicts.

Every rejection carries a reason code (`REJECTION_CODES`). The checks, in order:

- evidence: each quote must occur in the segment it cites, after normalizing only whitespace runs and curly
  quotes (and stripping paired wrapper quotes); offsets are recorded against the exact segment text, and the
  span must lie inside the chunk that owns it (neighbour lines are context, never evidence).
- times: a due date or validity bound is kept only when its expression occurs in the quotes and its value
  parses; unknown values stay unknown. Common relative expressions ("tomorrow", "in two weeks", "by Friday",
  "November 20") are recomputed from the meeting's local date and must agree. Validity is never "now".
- numbers: every number in the statement appears in the quotes, or is a component of a date derived above;
  an identifier such as `C2` or `Q3` counts as present when the quotes contain it, and ranges ("1 to 5",
  "1-5", "one to five") reduce to their end points.
- modality: hypothetical cues ("if we were to", "suppose") or negation cues ("won't", "not") in the quotes
  forbid `asserted`, except a non-commitment whose statement carries the negation itself.
- commitments: promise language in the quotes, no tentative cue in the promising quote, modality asserted,
  and each responsible person is the speaker of a cited line, named in the quotes, or a collective "we".
- prior decisions: a statement the owner already marked wrong or dismissed is not proposed again.
- dedupe: same normalized statement with overlapping evidence (within a source, across chunk overlap).

Lexical cues are a precision guard on short quotes, not a language model; prompts ask for minimal quotes.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from ..schema import (
    CandidateAssertion,
    EntityRef,
    EvidenceRef,
    ExtractRequest,
    LinkProposal,
    Segment,
    TimeBound,
)
from .prompts import Enriched, Mention, TimeExpression, Verdict

REJECTION_CODES = (
    "unknown_segment", "quote_not_found", "evidence_outside_chunk", "invalid_time", "date_derivation_mismatch",
    "ungrounded_number", "hypothetical_as_asserted", "negation_as_asserted", "commitment_not_asserted",
    "no_promise_language", "tentative_commitment", "attribution_mismatch", "invalid", "not_entailed",
    "attribution_rejected", "unverified", "prior_decision", "duplicate", "limit",
)

QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "‛": "'", "“": '"', "”": '"',
                             " ": " "})
WRAPPERS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
WORDS = re.compile(r"[\w']+")

HYPOTHETICAL = re.compile(
    r"\b(if (?:we|i|you|they|he|she) were|were (?:we|i|you|they) to|hypothetically|suppose|supposing|imagine|"
    r"what if|in theory|let's say|say we)\b", re.IGNORECASE)
NEGATION = re.compile(
    r"\b(won't|will not|wouldn't|would not|not going to|don't|do not|doesn't|does not|didn't|did not|can't|"
    r"cannot|can not|never|no longer|isn't|is not|aren't|are not|wasn't|weren't|shouldn't|decided against|not)\b",
    re.IGNORECASE)
PROMISE = re.compile(
    r"\b(i'll|i will|i'm going to|i am going to|i can|i'd be happy to|happy to|let me|we'll|we will|"
    r"we're going to|we are going to|will do|on it|i've got (?:it|this)|leave it with me|i promise|i commit|"
    r"count me in|sure|yes|yeah|okay|ok|absolutely|definitely|i'll take|i'll handle|action|agreed)\b", re.IGNORECASE)
TENTATIVE = re.compile(r"\b(maybe|might|perhaps|possibly|probably|could|should|if|unless|try to)\b", re.IGNORECASE)
COLLECTIVE = {"we", "us", "the team", "team", "everyone", "all of us", "both of us"}
COLLECTIVE_CUE = re.compile(r"\b(we|we'll|we're|us|team|everyone|all of us|both of us)\b", re.IGNORECASE)
UNKNOWN = {"unknown speaker", "unknown", "someone"}
NOW = {"now", "right now", "currently", "at the moment", "as of now", "as of today", "today"}

UNITS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])}
TENS = {w: 10 * i for i, w in enumerate(["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]) if i > 1}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000, "k": 1000,
          "m": 1_000_000, "bn": 1_000_000_000}
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
NUMBER_WORDS = re.compile(
    r"\b(?:(?:" + "|".join([*UNITS, *TENS, "hundred", "thousand", "million", "billion", "dozen"]) +
    r")(?:[\s-]+(?:and[\s-]+)?)?)+\b", re.IGNORECASE)

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
COUNT = r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|couple of)"
RELATIVE = re.compile(rf"\b(?:in|within|after)\s+{COUNT}\s+(day|week|month)s?\b|\b{COUNT}\s+(day|week|month)s?\s+"
                      r"from\s+(?:now|today)\b", re.IGNORECASE)
WEEKDAY = re.compile(r"\b(next\s+|last\s+)?(" + "|".join(WEEKDAYS) + r")\b", re.IGNORECASE)
MONTH_DAY = re.compile(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?|"
                       r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(MONTHS) + r")[a-z]*\b(?:,?\s+(\d{4}))?",
                       re.IGNORECASE)
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


@dataclass(frozen=True)
class Rejection:
    code: str
    statement: str
    detail: str = ""
    segment_seqs: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "statement": self.statement, "detail": self.detail,
                "segment_seqs": list(self.segment_seqs)}


@dataclass
class Draft:
    """An item that passed the deterministic checks. `assertion` is None for suggestions."""

    kind: str
    statement: str
    evidence: list[EvidenceRef]
    assertion: CandidateAssertion | None = None
    responsible: list[EntityRef] = field(default_factory=list)
    due: TimeBound | None = None
    links: list[LinkProposal] = field(default_factory=list)
    derivations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def seqs(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(e.segment_seq for e in self.evidence))

    def reject(self, code: str, detail: str = "") -> Rejection:
        return Rejection(code, self.statement, detail, self.seqs)


@dataclass(frozen=True)
class SourceContext:
    segments: dict[int, Segment]
    local_date: date
    zone: ZoneInfo
    candidates: dict[str, EntityRef]  # prompt id ("actor:act_1") -> resolved ref
    candidate_names: dict[str, list[str]]
    speaker_names: dict[str, set[str]]  # speaker label -> lowercased names it stands for
    blocked: frozenset[str]  # normalized statements of prior wrong/dismiss decisions

    @classmethod
    def from_request(cls, request: ExtractRequest) -> "SourceContext":
        zone = ZoneInfo(request.timezone or "UTC")
        names: dict[str, set[str]] = {}
        for p in request.participants:
            if p.speaker_label:
                names.setdefault(p.speaker_label, {p.speaker_label.casefold()}).update(
                    n.casefold() for n in (p.name, p.email and p.email.split("@")[0]) if n)
        for s in request.segments:
            if s.speaker_label:
                names.setdefault(s.speaker_label, {s.speaker_label.casefold()})
        return cls(
            segments={s.seq: s for s in request.segments},
            local_date=request.occurred_at.astimezone(zone).date(), zone=zone,
            candidates={candidate_key(c.ref): c.ref for c in request.candidates},
            candidate_names={candidate_key(c.ref): [c.name, *c.aliases] for c in request.candidates},
            speaker_names=names,
            blocked=frozenset(normalized(d.statement) for d in request.decisions
                              if d.statement and d.operation in ("wrong", "dismiss")))


def candidate_key(ref: EntityRef) -> str:
    return f"{ref.entity_type}:{ref.entity_id}"


def normalized(text: str) -> str:
    return " ".join(WORDS.findall(text.translate(QUOTE_CHARS).casefold()))


# ---------------------------------------------------------------- evidence


def _normal_map(text: str) -> tuple[str, list[int]]:
    """`text` with curly quotes straightened and whitespace runs collapsed, plus each output char's index."""
    out, index, previous_space = [], [], False
    for i, ch in enumerate(text.translate(QUOTE_CHARS)):
        if ch.isspace():
            if previous_space:
                continue
            ch, previous_space = " ", True
        else:
            previous_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def locate(segment_text: str, quote: str, owned: Callable[[int, int], bool] = lambda a, b: True
           ) -> tuple[int, int] | None:
    """Offsets of `quote` in `segment_text` (exact, else whitespace/quote-normalized), preferring an owned span."""
    quote = quote.strip()
    for left, right in WRAPPERS:
        if len(quote) > 2 and quote.startswith(left) and quote.endswith(right) and quote not in segment_text:
            quote = quote[1:-1].strip()
    hits = [(m.start(), m.start() + len(quote)) for m in re.finditer(re.escape(quote), segment_text)] if quote else []
    if not hits:
        text, index = _normal_map(segment_text)
        needle, _ = _normal_map(quote)
        needle = needle.strip()
        if needle:
            hits = [(index[m.start()], index[m.start() + len(needle) - 1] + 1)
                    for m in re.finditer(re.escape(needle), text)]
    return next((h for h in hits if owned(*h)), hits[0] if hits else None)


ELLIPSIS = re.compile(r"\.{3,}|\u2026")
MIN_FRAGMENT_WORDS = 3
FRAGMENT_NEIGHBOURS = (0, 1, -1, 2, -2)  # the cited line first, then lines a ref off by one or two would mean


def fragments(quote: str) -> list[str]:
    """The contiguous pieces of a quote stitched together with "..." or "…", trimmed, without pieces under
    MIN_FRAGMENT_WORDS words (a dangling "so" or "and")."""
    pieces = (p.strip().strip(",;:-\u2013\u2014").strip() for p in ELLIPSIS.split(quote))
    return [p for p in pieces if len(p.split()) >= MIN_FRAGMENT_WORDS]


def _locate_fragment(seq: int, fragment: str, ctx: SourceContext,
                     owns: Callable[[int, int, int], bool]) -> tuple[int, tuple[int, int]] | None:
    """A fragment in the cited segment, else in a neighbour within two lines, as (segment, span)."""
    for offset in FRAGMENT_NEIGHBOURS:
        segment = ctx.segments.get(seq + offset)
        if segment is None:
            continue
        for needle in dict.fromkeys((fragment, fragment.rstrip(".!?,;:").strip())):
            span = needle and locate(segment.text, needle, lambda a, b, s=seq + offset: owns(s, a, b))
            if span and owns(seq + offset, *span):
                return seq + offset, span
    return None


def ground_evidence(quotes: Iterable[tuple[int, str]], ctx: SourceContext,
                    owns: Callable[[int, int, int], bool]) -> tuple[list[EvidenceRef], str | None, str]:
    """Each quote as an exact span of its segment. A quote that is not one span but contiguous pieces joined with
    "..." or "…" grounds piece by piece (in the cited segment or up to two lines away), one span per piece in
    order; it is refused when any piece is found nowhere."""
    refs, seen = [], set()

    def add(seq: int, span: tuple[int, int]) -> None:
        if (seq, *span) not in seen:
            seen.add((seq, *span))
            text = ctx.segments[seq].text
            refs.append(EvidenceRef(segment_seq=seq, quote=text[span[0]:span[1]], char_start=span[0],
                                    char_end=span[1]))

    for seq, quote in quotes:
        segment = ctx.segments.get(seq)
        if segment is None:
            return [], "unknown_segment", f"segment {seq}"
        span = locate(segment.text, quote, lambda a, b, seq=seq: owns(seq, a, b))
        if span is None and ELLIPSIS.search(quote):
            pieces = fragments(quote)
            found = [_locate_fragment(seq, piece, ctx, owns) for piece in pieces]
            if not pieces or None in found:
                missing = next((p for p, f in zip(pieces, found, strict=True) if f is None), quote)
                return [], "quote_not_found", f"segment {seq}: {missing[:120]!r}"
            for where, piece_span in found:
                add(where, piece_span)
            continue
        if span is None:
            return [], "quote_not_found", f"segment {seq}: {quote[:120]!r}"
        if not owns(seq, *span):
            return [], "evidence_outside_chunk", f"segment {seq} is context for this chunk"
        add(seq, span)
    return refs, None, ""


# ---------------------------------------------------------------- numbers and dates


def _decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def _word_value(phrase: str) -> Decimal | None:
    total, current, seen = 0, 0, False
    for word in re.findall(r"[a-z]+", phrase.lower()):
        if word in UNITS:
            current += UNITS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "dozen":
            current = (current or 1) * 12
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in SCALES:
            total += (current or 1) * SCALES[word]
            current = 0
        else:
            continue
        seen = True
    return Decimal(total + current) if seen else None


def numbers(text: str) -> set[Decimal]:
    """Every number in `text`: digit tokens (and each digit run, so dates yield their parts), scaled digits
    ("40k", "1.5 million") and number-word phrases ("forty-two")."""
    text = text.translate(QUOTE_CHARS)
    found: set[Decimal] = set()
    for match in NUMBER.finditer(text):
        value = _decimal(match.group())
        if value is None:
            continue
        found.add(value)
        found.update(Decimal(part) for part in re.findall(r"\d+", match.group()))
        scale = re.match(r"\s*(k|m|bn|thousand|million|billion)\b", text[match.end():], re.IGNORECASE)
        if scale:
            found.add(value * SCALES[scale.group(1).lower()])
    for match in NUMBER_WORDS.finditer(text):
        value = _word_value(match.group())
        if value is not None:
            found.add(value)
    return {v.normalize() for v in found}


# An identifier is a letter prefix immediately followed by digits (C2, Q1, P0, v2): one token, never a number.
IDENTIFIER = re.compile(r"(?<![\w'])[A-Za-z]+\d+[A-Za-z\d]*(?![\w])")
# Digits immediately followed by letters (5k, 3x, 6th): grounded verbatim, else by the number they carry.
DIGIT_TOKEN = re.compile(r"(?<![\w'.,])\d+[A-Za-z]+(?![\w])")
JOINED = re.compile(r"(?<=\w)\s*[-/\u2010-\u2014]\s*(?=\w)")


def split_joined(text: str) -> str:
    """Hyphen-, dash- and slash-joined forms as separate words: "1-to-5", "1-5" and "1/5" are 1, to, 5."""
    return JOINED.sub(" ", text.translate(QUOTE_CHARS))


def strip_identifiers(statement: str, quotes: str) -> str:
    """`statement` without the alphanumeric identifiers (C2, Q3, v2) that `quotes` contains, so their digits are
    not read as numbers to ground."""
    text = quotes.translate(QUOTE_CHARS).casefold()
    return IDENTIFIER.sub(lambda m: " " if _word_in(m.group().casefold(), text) else m.group(), statement)


def ungrounded_tokens(statement: str, text: str, derived: Sequence[date | datetime]) -> list[str]:
    """For action items, grounded against the whole meeting: identifiers (C2, Q1, P0, v2) are whole tokens that
    must occur in `text`, never bare numbers; joined forms ("1-to-5", "1-5", "1/5") are their component words;
    digit tokens (5k, 3x) occur verbatim or carry a number `text` has; every other number must occur in `text`
    outside its identifiers (or be part of a derived date). Returns what is missing."""
    statement, text = split_joined(statement), split_joined(text)
    folded = text.casefold()
    missing = sorted({m.group() for m in IDENTIFIER.finditer(statement)
                      if not _word_in(m.group().casefold(), folded)}, key=str.casefold)
    allowed = numbers(IDENTIFIER.sub(" ", text))
    for value in derived:
        allowed |= {Decimal(value.year), Decimal(value.month), Decimal(value.day)}
        if isinstance(value, datetime):
            allowed |= {Decimal(value.hour), Decimal(value.minute), Decimal(value.hour % 12 or 12)}
    rest = IDENTIFIER.sub(" ", statement)
    for m in DIGIT_TOKEN.finditer(rest):  # "5k": verbatim, or any number it carries ("5,000", "five thousand")
        if _word_in(m.group().casefold(), folded) or numbers(m.group()) & allowed:
            rest = rest.replace(m.group(), " ", 1)
        else:
            missing.append(m.group())
            rest = rest.replace(m.group(), " ", 1)
    return missing + [format(n, "f") for n in sorted(numbers(rest) - allowed)]


def ungrounded_numbers(statement: str, quotes: str, derived: Sequence[date | datetime]) -> set[Decimal]:
    statement = strip_identifiers(statement, quotes)
    allowed = numbers(quotes)
    for value in derived:
        allowed |= {Decimal(value.year), Decimal(value.month), Decimal(value.day)}
        if isinstance(value, datetime):
            allowed |= {Decimal(value.hour), Decimal(value.minute), Decimal(value.hour % 12 or 12)}
    return numbers(statement) - allowed


def _count(word: str) -> int:
    word = word.lower()
    if word.isdigit():
        return int(word)
    return 1 if word in ("a", "an") else 2 if "couple" in word else UNITS[word]


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year, month = day.year + month // 12, month % 12 + 1
    for d in (day.day, 30, 29, 28):
        try:
            return date(year, month, d)
        except ValueError:
            continue
    raise ValueError("unreachable")


def expected_date(expression: str, meeting: date) -> tuple[date | None, bool]:
    """(date, year_known) the expression denotes relative to the meeting's local date, when unambiguous."""
    text = expression.translate(QUOTE_CHARS).lower()
    if m := ISO_DATE.search(text):
        return date(int(m[1]), int(m[2]), int(m[3])), True
    if m := RELATIVE.search(text):
        count, unit = (m[1], m[2]) if m[1] else (m[3], m[4])
        n = _count(count)
        unit = unit.lower()
        return (_add_months(meeting, n) if unit == "month" else
                meeting + timedelta(days=n * (7 if unit == "week" else 1))), True
    if "day after tomorrow" in text:
        return meeting + timedelta(days=2), True
    for word, offset in (("tomorrow", 1), ("yesterday", -1), ("today", 0), ("tonight", 0)):
        if re.search(rf"\b{word}\b", text):
            return meeting + timedelta(days=offset), True
    if m := MONTH_DAY.search(text):
        month_name, day, year = (m[1], m[2], m[3]) if m[1] else (m[5], m[4], m[6])
        try:
            return date(int(year) if year else meeting.year, MONTHS.index(month_name[:3]) + 1, int(day)), bool(year)
        except ValueError:
            return None, False
    if (m := WEEKDAY.search(text)) and not m[1] and not re.search(r"\b(?:end of|week of)\b", text):
        ahead = (WEEKDAYS.index(m[2].lower()) - meeting.weekday()) % 7 or 7
        return meeting + timedelta(days=ahead), True
    return None, False


def _parse_value(value: str, zone: ZoneInfo, precision: str = "instant") -> datetime | None:
    """A model's time value as an aware datetime. A date-precision value (day and coarser) is the calendar date
    it names, at local midnight in the meeting's zone: "2026-11-04T00:00:00Z" for "today" is Nov 4 there, never
    midnight UTC shifted back to Nov 3."""
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) or (precision not in ("instant", "unknown")
                                                        and re.match(r"\d{4}-\d{2}-\d{2}(?:[T ]|$)", value)):
            return datetime.combine(date.fromisoformat(value[:10]), time(), zone)
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=zone)


def resolve_time(name: str, te: TimeExpression | None, quotes: str, ctx: SourceContext
                 ) -> tuple[TimeBound | None, dict[str, Any] | None, str | None]:
    """(bound or None, derivation record, rejection code). Unknown or ungrounded timing becomes None."""
    if te is None:
        return None, None, None
    record = {"field": name, "expression": te.expression, "value": te.value, "precision": te.precision,
              "derivation": te.derivation, "verified": False}
    expression = normalized(te.expression)
    if name == "valid_from" and expression in NOW:
        return None, {**record, "dropped": "validity is never now; occurred_at applies"}, None
    if not expression or expression not in normalized(quotes):
        return None, {**record, "dropped": "expression not in quotes"}, None
    if te.value is None or te.precision == "unknown":
        return None, {**record, "dropped": "unknown"}, None
    at = _parse_value(te.value, ctx.zone, te.precision)
    if at is None:
        return None, {**record, "dropped": "unparseable value"}, "invalid_time"
    expected, year_known = expected_date(te.expression, ctx.local_date)
    local = at.astimezone(ctx.zone).date()
    if expected is not None:
        if (local if year_known else local.replace(year=expected.year)) != expected:
            return None, {**record, "expected": expected.isoformat()}, "date_derivation_mismatch"
        record["verified"] = True
    return TimeBound(at=at, precision=te.precision, expression=te.expression[:200]), record, None


# ---------------------------------------------------------------- people


def _entity(mention: Mention | None, ctx: SourceContext) -> EntityRef | None:
    if mention is None:
        return None
    ref = ctx.candidates.get(mention.candidate_id or "")
    if ref is not None and ref.entity_type == mention.entity_type:
        return EntityRef(entity_type=ref.entity_type, entity_id=ref.entity_id, mention=mention.mention[:300])
    return EntityRef(entity_type=mention.entity_type, mention=mention.mention[:300])


def _names(ref: EntityRef, ctx: SourceContext) -> set[str]:
    names = {ref.mention} if ref.mention else set()
    if ref.entity_id:
        names.update(ctx.candidate_names.get(candidate_key(ref), []))
    return {n.casefold().strip() for n in names if n and n.strip()}


def _word_in(name: str, text: str) -> bool:
    return bool(re.search(rf"(?<![\w']){re.escape(name)}(?![\w])", text))


def attributed(ref: EntityRef, evidence: Sequence[EvidenceRef], ctx: SourceContext) -> bool:
    names = _names(ref, ctx)
    quotes = " ".join(e.quote for e in evidence).translate(QUOTE_CHARS).casefold()
    labels = [ctx.segments[e.segment_seq].speaker_label for e in evidence]
    if names & UNKNOWN:
        return any(label is None for label in labels)
    if names & COLLECTIVE:
        return bool(COLLECTIVE_CUE.search(quotes))
    speakers = set().union(*(ctx.speaker_names.get(label, set()) for label in labels if label))
    firsts = {n.split()[0] for n in names}
    if names & speakers or firsts & {s.split()[0] for s in speakers if s}:
        return True
    return any(_word_in(n, quotes) for n in names | firsts)


def promise_quotes(evidence: Sequence[EvidenceRef], responsible: Sequence[EntityRef], ctx: SourceContext
                   ) -> list[str]:
    """The quotes that carry promise language: generic cues, or a responsible name followed by will/to/'ll."""
    names = set().union(*(_names(r, ctx) for r in responsible)) if responsible else set()
    named = [re.compile(rf"\b{re.escape(n)}\b\s*(?:'ll|will|to|is going to|can|agreed to)\b", re.IGNORECASE)
             for n in names | {n.split()[0] for n in names} if n]
    out = []
    for e in evidence:
        quote = e.quote.translate(QUOTE_CHARS)
        if PROMISE.search(quote) or any(p.search(quote) for p in named):
            out.append(quote)
    return out


# ---------------------------------------------------------------- the deterministic pass


def check_item(item: Enriched, ctx: SourceContext, owns: Callable[[int, int, int], bool]) -> Draft | Rejection:
    statement = item.statement.strip()
    evidence, code, detail = ground_evidence(((q.segment_seq, q.quote) for q in item.evidence), ctx, owns)
    if code:
        return Rejection(code, statement, detail, tuple(q.segment_seq for q in item.evidence))
    draft = Draft(kind=item.kind, statement=statement, evidence=evidence)
    quotes = " ".join(e.quote for e in evidence)
    commitment = item.kind in ("commitment", "suggestion")

    bounds = {}
    for name in ("due", "valid_from", "valid_to"):
        te = getattr(item, name) if name != "due" or commitment else None
        bound, record, code = resolve_time(name, te, quotes, ctx)
        if code:
            return draft.reject(code, f"{name}: {te.expression!r} -> {te.value}")
        if record:
            draft.derivations.append(record)
        bounds[name] = bound
    missing = ungrounded_numbers(statement, quotes, [b.at.astimezone(ctx.zone) for b in bounds.values() if b])
    if missing:
        return draft.reject("ungrounded_number", ", ".join(sorted(str(n) for n in missing)))

    modality = item.modality
    if HYPOTHETICAL.search(quotes.translate(QUOTE_CHARS)) and modality == "asserted":
        return draft.reject("hypothetical_as_asserted")
    responsible = [r for r in (_entity(m, ctx) for m in item.responsible) if r] if commitment else []
    if item.kind == "commitment":
        if modality != "asserted":
            return draft.reject("commitment_not_asserted", modality)
        promising = promise_quotes(evidence, responsible, ctx)
        if not promising:
            return draft.reject("no_promise_language")
        if any(TENTATIVE.search(q) or NEGATION.search(q) for q in promising):
            return draft.reject("tentative_commitment")
        if not responsible:
            return draft.reject("attribution_mismatch", "no responsible person")
        wrong = [r.mention or r.entity_id for r in responsible if not attributed(r, evidence, ctx)]
        if wrong:
            return draft.reject("attribution_mismatch", ", ".join(str(w) for w in wrong))
    elif modality == "asserted" and NEGATION.search(quotes.translate(QUOTE_CHARS)) \
            and not NEGATION.search(statement.translate(QUOTE_CHARS)):
        return draft.reject("negation_as_asserted")

    if normalized(statement) in ctx.blocked:
        return draft.reject("prior_decision")
    draft.responsible, draft.due = responsible, bounds["due"]
    if item.kind == "suggestion":
        return draft
    try:
        draft.assertion = CandidateAssertion(
            kind=item.kind, statement=statement[:2000], subject=_entity(item.subject, ctx),
            object=_entity(item.object, ctx), key=(item.key or None) and item.key[:100], modality=modality,
            inferred=item.inferred, valid_from=bounds["valid_from"], valid_to=bounds["valid_to"],
            responsible=responsible if item.kind == "commitment" else [],
            recipient=_entity(item.recipient, ctx) if item.kind == "commitment" else None,
            due=bounds["due"] if item.kind == "commitment" else None, evidence=evidence,
            confidence=item.confidence)
    except ValidationError as error:
        return draft.reject("invalid", error.errors()[0]["msg"])
    draft.links = links_for(draft.assertion, evidence)
    return draft


def links_for(assertion: CandidateAssertion, evidence: list[EvidenceRef]) -> list[LinkProposal]:
    """Mentions the model tied to a listed candidate, as inferred links for the linker to confirm or replace."""
    refs = [assertion.subject, assertion.object, assertion.recipient, *assertion.responsible]
    return [LinkProposal(mention=r.mention or r.entity_id, candidate=EntityRef(entity_type=r.entity_type,
                                                                                entity_id=r.entity_id),
                         status="inferred", evidence=evidence)
            for r in refs if r is not None and r.entity_id]


# ---------------------------------------------------------------- verdicts and dedupe


def apply_verdicts(drafts: Sequence[Draft], verdicts: Sequence[Verdict]) -> tuple[list[Draft], list[Rejection]]:
    by_item = {v.item: v for v in verdicts}
    kept, rejected = [], []
    for i, draft in enumerate(drafts):
        verdict = by_item.get(i)
        if verdict is None:
            rejected.append(draft.reject("unverified"))
            continue
        if not verdict.entailed:
            rejected.append(draft.reject("not_entailed", verdict.reason))
            continue
        if draft.kind == "commitment" and not verdict.attribution_ok:
            rejected.append(draft.reject("attribution_rejected", verdict.reason))
            continue
        current = draft.assertion.modality if draft.assertion else "proposed"
        if verdict.modality != current and current == "asserted":
            if draft.kind == "commitment":
                rejected.append(draft.reject("commitment_not_asserted", f"verifier: {verdict.modality}"))
                continue
            draft.assertion = draft.assertion.model_copy(update={"modality": verdict.modality})
            draft.notes.append(f"modality {current} -> {verdict.modality}: {verdict.reason}")
        kept.append(draft)
    return kept, rejected


def _overlaps(a: Sequence[EvidenceRef], b: Sequence[EvidenceRef]) -> bool:
    return any(x.segment_seq == y.segment_seq and x.char_start < y.char_end and y.char_start < x.char_end
               for x in a for y in b)


def dedupe(drafts: Iterable[Draft]) -> tuple[list[Draft], list[Rejection]]:
    kept: list[Draft] = []
    rejected = []
    for draft in drafts:
        key = (draft.kind == "suggestion", normalized(draft.statement))
        if any((k.kind == "suggestion", normalized(k.statement)) == key and _overlaps(k.evidence, draft.evidence)
               for k in kept):
            rejected.append(draft.reject("duplicate"))
        else:
            kept.append(draft)
    return kept, rejected
