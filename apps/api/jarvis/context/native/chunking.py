"""Split a source revision's segments into extraction chunks.

A chunk owns consecutive spans (whole segments, or exact slices of a segment too long for one chunk) and carries
up to `neighbour_tokens` of the surrounding lines as read-only context. Evidence may cite only owned spans, so
every character is owned by exactly one chunk. Chunks close at a speaker change when one exists in their second
half; a turn longer than `target_tokens` splits at sentence boundaries (then whitespace, then characters) with
its offsets kept. Token counts use `model_map.estimate_tokens` (about four characters per token): a cheap,
deterministic estimate for sizing, never for billing.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ...model_map import estimate_tokens
from ..schema import Chunk, Segment

SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Span:
    """`text == segment.text[start:end]`; offsets are within the segment."""

    seq: int
    start: int
    end: int
    text: str
    speaker: str | None
    kind: str
    part: int = 0  # 0: the whole segment; 1..n: slices of a split segment

    def line(self) -> str:
        who = "Private note" if self.kind == "private_note" else (self.speaker or "Unknown speaker")
        cont = " (continued)" if self.part > 1 else ""
        return f"[{self.seq}] {who}{cont}: {self.text}"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.line())


@dataclass(frozen=True)
class TextChunk:
    seq: int
    spans: tuple[Span, ...]
    before: tuple[Span, ...]  # context only
    after: tuple[Span, ...]  # context only
    token_count: int

    @property
    def segment_seqs(self) -> list[int]:
        return list(dict.fromkeys(s.seq for s in self.spans))

    def owns(self, seq: int, start: int, end: int) -> bool:
        return any(s.seq == seq and s.start <= start and end <= s.end for s in self.spans)

    def text(self) -> str:
        return "\n".join(s.line() for s in self.spans)

    def contract(self) -> Chunk:
        return Chunk(seq=self.seq, segment_start=self.spans[0].seq, segment_end=self.spans[-1].seq,
                     token_count=self.token_count, text=self.text())


def _cut(text: str, start: int, limit: int) -> int:
    """End offset of the longest piece of `text[start:]` within `limit` chars: at a sentence end, else at
    whitespace, else exactly at the limit."""
    if len(text) - start <= limit:
        return len(text)
    window = text[start:start + limit]
    ends = [m.end() for m in SENTENCE_END.finditer(window)]
    if ends and ends[-1] > limit // 4:
        return start + ends[-1]
    space = window.rfind(" ")
    return start + space + 1 if space > limit // 4 else start + limit


def split_segment(segment: Segment, target_tokens: int) -> list[Span]:
    def span(start: int, end: int, part: int) -> Span:
        return Span(segment.seq, start, end, segment.text[start:end], segment.speaker_label, segment.kind, part)

    whole = span(0, len(segment.text), 0)
    if whole.tokens <= target_tokens:
        return [whole]
    prefix = len(whole.line()) - len(segment.text) + len(" (continued)")
    limit = max(CHARS_PER_TOKEN, (target_tokens * CHARS_PER_TOKEN) - prefix)
    spans, start = [], 0
    while start < len(segment.text):
        end = _cut(segment.text, start, limit)
        spans.append(span(start, end, len(spans) + 1))
        start = end
    return spans


def _boundary(units: list[Span], start: int, end: int) -> int:
    """Where to close a chunk holding units[start:end] when units[end] does not fit: the last speaker change in
    the second half of the chunk, else `end`."""
    for i in range(end, start + (end - start) // 2, -1):
        if i < len(units) and (units[i].speaker != units[i - 1].speaker or units[i].kind != units[i - 1].kind):
            return i
    return end


def _context(units: Sequence[Span], budget: int, *, reverse: bool) -> tuple[Span, ...]:
    taken = []
    for unit in units:
        if budget <= 0:
            break
        if unit.tokens > budget:
            chars = budget * CHARS_PER_TOKEN
            if reverse:
                start = max(unit.start, unit.end - chars)
                unit = Span(unit.seq, start, unit.end, unit.text[start - unit.start:], unit.speaker, unit.kind,
                            max(unit.part, 2) if start > 0 else unit.part)
            else:
                end = min(unit.end, unit.start + chars)
                unit = Span(unit.seq, unit.start, end, unit.text[:end - unit.start], unit.speaker, unit.kind,
                            unit.part)
        taken.append(unit)
        budget -= unit.tokens
    return tuple(reversed(taken)) if reverse else tuple(taken)


def chunk_segments(segments: Sequence[Segment], target_tokens: int = 5000,
                   neighbour_tokens: int = 500) -> list[TextChunk]:
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    units = [span for segment in sorted(segments, key=lambda s: s.seq)
             for span in split_segment(segment, target_tokens)]
    ranges, start = [], 0
    while start < len(units):
        end, size = start, 0
        while end < len(units) and (end == start or size + units[end].tokens <= target_tokens):
            size += units[end].tokens
            end += 1
        if end < len(units):
            end = _boundary(units, start, end)
        ranges.append((start, end))
        start = end
    return [TextChunk(seq=i, spans=tuple(units[a:b]),
                      before=_context(units[a - 1::-1] if a else [], neighbour_tokens, reverse=True),
                      after=_context(units[b:], neighbour_tokens, reverse=False),
                      token_count=sum(u.tokens for u in units[a:b]))
            for i, (a, b) in enumerate(ranges)]
