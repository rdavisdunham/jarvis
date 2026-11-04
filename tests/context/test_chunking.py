"""Chunking: complete exclusive ownership, size bounds, speaker boundaries, exact offsets, read-only neighbours."""

from jarvis.context.native.chunking import chunk_segments
from jarvis.context.schema import Chunk, Segment

from .fixtures.extraction import long_meeting


def owned_text(chunks, seq):
    return "".join(s.text for c in chunks for s in c.spans if s.seq == seq)


def test_short_meeting_is_one_chunk_without_neighbours(envelope):
    [chunk] = chunk_segments(envelope.segments)
    assert chunk.segment_seqs == [0, 1, 2, 3, 4] and not chunk.before and not chunk.after
    assert chunk.contract() == Chunk(seq=0, segment_start=0, segment_end=4, token_count=chunk.token_count,
                                     text=chunk.text())
    assert chunk.text().splitlines()[1] == "[1] Priya: The pilot covers 40 clinics in the first wave."


def test_long_meeting_chunks_own_every_character_once_within_bounds():
    segments = long_meeting.segments()
    chunks = chunk_segments(segments)
    assert len(chunks) >= 6
    assert all(c.token_count <= 5000 for c in chunks)
    assert all(owned_text(chunks, s.seq) == s.text for s in segments)
    spans = [(s.seq, s.start) for c in chunks for s in c.spans]
    assert spans == sorted(spans) and len(spans) == len(set(spans))
    assert [c.seq for c in chunks] == list(range(len(chunks)))
    assert chunk_segments(segments) == chunks  # deterministic


def test_oversized_turn_splits_at_sentence_boundaries_with_exact_offsets():
    segments = long_meeting.segments()
    long = segments[long_meeting.LONG_AT]
    chunks = chunk_segments(segments)
    parts = [s for c in chunks for s in c.spans if s.seq == long.seq]
    assert len(parts) >= 3 and [p.part for p in parts] == list(range(1, len(parts) + 1))
    assert all(long.text[p.start:p.end] == p.text for p in parts)
    assert all(p.text.rstrip().endswith(".") for p in parts)
    assert [p.start for p in parts[1:]] == [p.end for p in parts[:-1]]
    assert "(continued)" in parts[1].line() and "(continued)" not in parts[0].line()
    # a split segment appears in several chunks; each chunk owns only its slice
    holder = next(c for c in chunks if parts[1] in c.spans)
    assert holder.owns(long.seq, parts[1].start, parts[1].end)
    assert not holder.owns(long.seq, parts[0].start, parts[0].end)


def test_text_without_sentences_splits_at_whitespace_then_characters():
    words = Segment(seq=0, speaker_label="A", text=" ".join(["word"] * 400))
    blob = Segment(seq=1, speaker_label="A", text="x" * 1000)
    chunks = chunk_segments([words, blob], target_tokens=100, neighbour_tokens=0)
    assert owned_text(chunks, 0) == words.text and owned_text(chunks, 1) == blob.text
    assert all(s.tokens <= 100 for c in chunks for s in c.spans)
    assert all(s.text.endswith(" ") for c in chunks for s in c.spans if s.seq == 0 and s.end < len(words.text))


def test_chunks_prefer_closing_at_a_speaker_change():
    lines = [("A", 30), ("A", 30), ("B", 30), ("B", 30), ("B", 30), ("C", 30)]
    segments = [Segment(seq=i, speaker_label=who, text="w" * (n * 4 - 12)) for i, (who, n) in enumerate(lines)]
    chunks = chunk_segments(segments, target_tokens=100, neighbour_tokens=0)
    # the first chunk could hold three lines; it stops at the A->B change instead of splitting B's run
    assert [c.segment_seqs for c in chunks] == [[0, 1], [2, 3, 4], [5]]


def test_neighbours_are_bounded_context_never_owned():
    chunks = chunk_segments(long_meeting.segments(), neighbour_tokens=500)
    for prev, chunk, nxt in zip(chunks, chunks[1:], chunks[2:], strict=False):
        assert chunk.before and chunk.after
        assert sum(s.tokens for s in chunk.before) <= 510 and sum(s.tokens for s in chunk.after) <= 510
        assert chunk.before[-1].seq == prev.spans[-1].seq and chunk.after[0].seq == nxt.spans[0].seq
        assert not any(chunk.owns(s.seq, s.start, s.end) for s in (*chunk.before, *chunk.after))
    assert not chunks[0].before and not chunks[-1].after
    assert all(not c.before and not c.after for c in chunk_segments(long_meeting.segments(), neighbour_tokens=0))
