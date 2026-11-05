"""A deterministic long synthetic meeting (about 90 minutes) and a scripted stand-in for the extraction model.

The transcript is data. It rotates four named speakers plus an unattributed line now and then, plants a
commitment ("I'll ... by Friday.") every 23rd line and a decision ("We decided to ...") every 31st, and holds one
monologue far longer than a chunk so the chunker must split it. `ScriptedModel` answers the chunk passes and the
whole-meeting action pass the way a well-behaved model would, so tests exercise chunking, validation, coverage and
cost end to end without a network.
"""

import json
import random
import re
from datetime import UTC, date, datetime, timedelta

from jarvis.context.schema import ExtractRequest, Participant, Segment
from jarvis.model_map import estimate_tokens

MEETING_AT = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)  # Tuesday 10:00 in New York
TIMEZONE = "America/New_York"
SPEAKERS = ["Davin", "Priya", "Marco", "Lena"]
SUBJECTS = ["the rollout plan", "the clinic onboarding", "the vendor contract", "the data migration",
            "the pilot budget", "the support rota", "the reporting dashboard", "the training sessions"]
OPINIONS = ["looks mostly on track", "needs another review pass", "depends on the vendor timeline",
            "came up in last week's call", "is the main thing the clinics ask about", "has a few open threads"]
FILLER = ["Right, that makes sense.", "Let me think about that for a second.", "Good point.",
          "I was going to say the same thing.", "Can you say more about that?", "Okay, moving on."]
TASKS = ["send the updated rollout plan", "draft the clinic welcome email", "book the training rooms",
         "review the vendor contract", "share the migration checklist", "update the pilot budget sheet"]
DECISIONS = ["keep the pilot on the current vendor", "run training sessions on Thursdays",
             "move reporting into the shared dashboard", "start onboarding with the northern clinics"]
COMMIT_EVERY, DECIDE_EVERY, LONG_AT, LINES = 23, 31, 400, 720


def _sentence(rng: random.Random) -> str:
    return f"I think {rng.choice(SUBJECTS)} {rng.choice(OPINIONS)}."


def segments(lines: int = LINES, seed: int = 11) -> list[Segment]:
    rng = random.Random(seed)
    out = []
    for i in range(lines):
        speaker = None if i % 69 == 30 else SPEAKERS[(i * 7 + i // 5) % len(SPEAKERS)]
        if i == LONG_AT:
            text = " ".join(_sentence(rng) for _ in range(900))  # ~50,000 characters: several chunks' worth
            text += f" I'll {TASKS[0]} by Friday."
        elif i % COMMIT_EVERY == 7:
            text = f"{rng.choice(FILLER)} I'll {TASKS[i % len(TASKS)]} by Friday."
        elif i % DECIDE_EVERY == 11:
            text = f"We decided to {DECISIONS[i % len(DECISIONS)]}."
        else:
            text = " ".join(rng.choice([_sentence(rng), rng.choice(FILLER)]) for _ in range(rng.randint(1, 3)))
        out.append(Segment(seq=i, speaker_label=speaker, text=text, start_ms=i * 7_500, end_ms=i * 7_500 + 7_000))
    return out


def _speaker(i: int) -> str:
    return "Unknown speaker" if i % 69 == 30 else SPEAKERS[(i * 7 + i // 5) % len(SPEAKERS)]


def follow_ups(lines: int = LINES) -> set[tuple[str, str]]:
    """(speaker, task) of each planted promise. The same person repeating the same promise is one follow-up."""
    return {(_speaker(i), TASKS[0] if i == LONG_AT else TASKS[i % len(TASKS)])
            for i in range(lines) if i % COMMIT_EVERY == 7 or i == LONG_AT}


def expected_counts(lines: int = LINES) -> tuple[int, int]:
    """(commitments, decisions) planted in the transcript. A promise on a line without a speaker is an unassigned
    follow-up, not a commitment."""
    commitments = sum(1 for who, _ in follow_ups(lines) if who != "Unknown speaker")
    decisions = sum(1 for i in range(lines) if i % DECIDE_EVERY == 11 and i % COMMIT_EVERY != 7 and i != LONG_AT)
    return commitments, decisions


def request(lines: int = LINES) -> ExtractRequest:
    return ExtractRequest(owner_id="davin", source_revision_id="srcrev_long", occurred_at=MEETING_AT,
                          timezone=TIMEZONE, title="Pilot rollout working session",
                          participants=[Participant(name=s, speaker_label=s) for s in SPEAKERS],
                          segments=segments(lines))


# ---------------------------------------------------------------- the scripted model

LINE = re.compile(r"^\[(\d+)\] (.+?)(?: \(continued\))?: (.*)$")
COMMIT = re.compile(r"I'll ([^.]+?) by (Friday)\.")
DECIDE = re.compile(r"We decided to ([^.]+)\.")
REASONING_TOKENS = 1000  # assumed hidden reasoning per call at low effort, so the cost estimate is not flattering


def _friday(meeting: date) -> date:
    return meeting + timedelta(days=(4 - meeting.weekday()) % 7 or 7)


class ScriptedModel:
    """Answers OpenAI Responses requests for the three passes. `fail` maps a pass name to the call numbers
    (1-based, per pass) that should return HTTP 500."""

    def __init__(self, fail: dict[str, set[int]] | None = None):
        self.calls: list[str] = []
        self.fail = fail or {}

    def __call__(self, body: dict) -> tuple[int, dict]:
        name = body["text"]["format"]["name"]
        self.calls.append(name)
        if self.calls.count(name) in self.fail.get(name, set()):
            return 500, {"error": {"message": "synthetic outage"}}
        user = json.loads(body["input"][1]["content"])
        output = getattr(self, name.removeprefix("context_"))(user)
        text = json.dumps(output)
        usage = {"input_tokens": estimate_tokens(json.dumps(body["input"])),
                 "output_tokens": estimate_tokens(text) + REASONING_TOKENS}
        return 200, {"id": f"resp_{len(self.calls)}", "status": "completed", "service_tier": "default",
                     "usage": usage, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    @staticmethod
    def _lines(user: dict) -> list[tuple[int, str, str]]:
        return [(int(m[1]), m[2], m[3]) for line in user["lines"].splitlines() if (m := LINE.match(line))]

    def detect(self, user: dict) -> dict:
        return {"candidates": [{"kind": "decision", "summary": m[0], "segment_seqs": [seq], "quote": m[0]}
                               for seq, _, text in self._lines(user) for m in DECIDE.finditer(text)]}

    def actions(self, user: dict) -> dict:
        meeting = date.fromisoformat(user["meeting_local_date"])
        return {"items": [
            {"evidence": [{"line": str(seq), "quote": m[0]}], "title": m[1][0].upper() + m[1][1:],
             "description": "", "owner": who, "owner_basis": "stated_promise", "recipient": None,
             "due": {"expression": "by Friday", "value": _friday(meeting).isoformat(), "precision": "day",
                     "derivation": "next Friday after the meeting date"},
             "status": "open", "kind": "action", "confidence": 0.9}
            for seq, who, text in self._lines(user) for m in COMMIT.finditer(text)]}

    def actions_verify(self, user: dict) -> dict:
        return {"verdicts": [{"item": it["item"], "progress_report": False, "real_follow_up": True, "owner_correct": True, "reason": ""}
                             for it in user["items"]]}

    def enrich(self, user: dict) -> dict:
        speakers = {seq: who for seq, who, _ in self._lines(user)}
        meeting = date.fromisoformat(user["meeting_local_date"])
        items = []
        for i, c in enumerate(user["candidates"]):
            seq, quote = c["segment_seqs"][0], c["quote"]
            item = {"candidate": i, "kind": c["kind"], "subject": None, "object": None, "modality": "asserted",
                    "inferred": False, "responsible": [], "responsibility": "none", "recipient": None,
                    "due": None, "valid_from": None, "valid_to": None, "key": None, "confidence": 0.9,
                    "evidence": [{"segment_seq": seq, "quote": quote}]}
            if c["kind"] == "commitment":
                task = COMMIT.search(quote)[1]
                who = speakers[seq]
                item.update(statement=f"{who} will {task} by Friday.", key="due_date", responsibility="speaker",
                            responsible=[{"entity_type": "actor", "mention": who, "candidate_id": None}],
                            due={"expression": "by Friday", "value": _friday(meeting).isoformat(),
                                 "precision": "day", "derivation": "next Friday after the meeting date"})
            else:
                item["statement"] = f"The team decided to {DECIDE.search(quote)[1]}."
            items.append(item)
        return {"items": items}

    def verify(self, user: dict) -> dict:
        return {"verdicts": [{"item": it["item"], "entailed": True, "modality": it["modality"],
                              "attribution_ok": True, "reason": ""} for it in user["items"]]}
