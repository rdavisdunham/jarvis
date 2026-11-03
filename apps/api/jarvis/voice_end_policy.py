"""Conservative, session-local ending authorization. Unknown meaning keeps listening.

Exact phrase inventories recognize a small command vocabulary; these are not a
semantic classifier. Models may propose intent, but cannot bypass freshness or
turn ownership. All durable metadata lives in existing encrypted payloads.
"""
import re
import time

from .assistant_identity import SPOKEN_NAMES

from sqlalchemy import select

from .models import AgentWork, Job, VoiceInbox, uid, now
from .domain import advisory
from .work_crypto import seal, unseal


def normalize(text):
    return " ".join(re.sub(r"[^a-z\s]", " ", text.lower().replace("’", "'").replace("'", "")).split())


EXPLICIT = {
    "goodbye", "good bye", "bye", "bye bye", "goodnight", "good night",
    "see you later", "talk to you later", "end voice", "stop voice", "end the voice session",
    "end our conversation", "end the conversation", "stop listening", "stop listening now",
    "im done talking", "i am done talking", "were finished talking", "thanks goodbye", "thank you goodbye", "that will be all goodbye", "im done with this conversation",
}
AMBIGUOUS = {"im done", "i am done", "thats all", "that is all", "im good", "i am good",
             "thats it for now", "thatll do for now", "were good for now", "nothing else",
             "i dont need anything else", "thats all for now"}
QUESTIONS = {"anything else": "more", "anything else i can help you with": "more",
             "will that be all": "done", "will that be all for now": "done"}
YES = {"yes", "yeah", "yep", "yes thanks", "yes thank you", "yep thanks"}
NO = {"no", "nope", "no thanks", "no thank you", "nope thank you"}


def command(text):
    if '"' in text or text.strip().startswith(("'", "“", "‘")):
        return ""
    value = normalize(text)
    for prefix in (*[name + " " for name in SPOKEN_NAMES], "okay ", "ok ", "please ", "you can "):
        if value.startswith(prefix):
            value = value[len(prefix):]
    for suffix in (*[" " + name for name in SPOKEN_NAMES], " thank you", " thanks", " please"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    return value


def closing_question(text):
    # Only the actual complete question approved by the controller can arm context.
    return QUESTIONS.get(normalize(text)) if text.strip().endswith("?") and text.count("?") == 1 else None


def decide(text, *, question=None, age=0, busy=False, pending=False, final_action_accepted=False):
    value = command(text)
    if value in EXPLICIT:
        return "explicit"
    if pending or busy:
        return "keep"
    if question and age <= 60:
        reply = normalize(text)
        if (question == "more" and reply in NO) or (question == "done" and reply in YES):
            return "confirm"
    # Only a specific trailing instruction; the worker must have accepted the action first.
    if final_action_accepted and normalize(text).endswith(" then end voice"):
        return "final_action"
    return "offer" if value in AMBIGUOUS else "keep"


class EndPolicy:
    def __init__(self):
        self.text, self.assistant_text = "", ""
        self.last_end = None
        self.assistant_since = False
        self.reply = None
        self.reply_age = 0
        self.observed = None
        self.approved = None
        self.last_offer = float("-inf")
        self.offer_count = 0
        self.offered_sequence = None
        self.sequence = 0
        self.utterance = 0
        self.task_question = False
        self.task_reply = False

    def input(self, delta, start, end, clock=None):
        clock = time.monotonic() if clock is None else clock
        if self.last_end is None or self.assistant_since or start - self.last_end > 750:
            self.text = ""
            self.utterance += 1
            self.task_reply = self.task_question
            self.task_question = False
            self.reply = self.observed[0] if self.observed else None
            self.reply_age = clock - self.observed[1] if self.observed else 0
            self.observed = self.approved = None
        self.assistant_since = False
        self.assistant_text = ""
        self.text += delta
        self.last_end = end
        self.sequence += 1
        return {"text": self.text, "question": self.reply, "age": self.reply_age, "pending": self.task_reply}

    def assistant(self, delta, clock=None):
        clock = time.monotonic() if clock is None else clock
        self.assistant_since = True
        self.assistant_text += delta
        if "?" in self.assistant_text:
            kind = closing_question(self.assistant_text)
            self.observed = (kind, clock) if self.approved and kind == self.approved else None
            self.task_question = not bool(self.observed)

    def offer(self, *, busy=False, pending=False, unaccepted=False, clock=None):
        clock = time.monotonic() if clock is None else clock
        if (busy or pending or self.task_reply or unaccepted or not self.sequence or self.utterance == self.offered_sequence
                or clock - self.last_offer < 180 or decide(self.text) != "offer"):
            return None
        self.offered_sequence, self.last_offer = self.utterance, clock
        self.approved = "done" if self.offer_count % 2 == 0 else "more"
        self.offer_count += 1
        self.assistant_text = ""
        return "Will that be all for now?" if self.approved == "done" else "Anything else?"


def has_outstanding(db, inbox, exclude=None):
    return db.scalar(select(AgentWork.id).join(Job).where(
        AgentWork.voice_session_id == inbox.id, AgentWork.owner_id == inbox.owner_id,
        AgentWork.account_id == inbox.account_id, AgentWork.id != (exclude or ""),
        Job.status.in_(["queued", "dispatched", "running", "needs_input", "waiting_sync", "retrying"]),
        AgentWork.result["archived_at"].as_string().is_(None)).limit(1)) is not None


def propose(row, *, final_action_accepted=False):
    from .db import session_scope
    from .access import assert_current
    from .domain import DomainError
    from .latency import mark
    outcome = {"status": "keep_listening", "voice_ended": False}
    accepted_input = unseal(row.input_ciphertext)
    source = accepted_input.get("voice_source")
    if not source:
        return outcome
    try:
        assert_current(row.owner_id, row.device_id)
    except DomainError:
        return outcome
    with session_scope() as db:
        advisory(db, "voice-inbox:" + row.voice_session_id)
        inbox = db.get(VoiceInbox, row.voice_session_id, populate_existing=True)
        if (not inbox or inbox.closed or (inbox.owner_id, inbox.account_id, inbox.device_id, inbox.conversation_id)
                != (row.owner_id, row.account_id, row.device_id, row.conversation_id)):
            return outcome
        data = unseal(inbox.content_ciphertext)
        if data.get("user_marker") != source.get("marker"):
            mark("voice_end_rejected", row.id, outcome="stale")
            return outcome
        ctx = source.get("context", {})
        text = ctx.get("text", "")
        resolved_final = (final_action_accepted and accepted_input.get("clarification_answer")
                          and normalize(accepted_input.get("continuation_request", "")).endswith(" then end voice")
                          and normalize(text) not in {phrase + suffix for phrase in ("thanks", "thank you")
                                                     for suffix in ("", *[" " + name for name in SPOKEN_NAMES])})
        if resolved_final:
            text = accepted_input["continuation_request"]
        category = decide(text, question=ctx.get("question"), age=ctx.get("age", 61),
                          pending=(ctx.get("pending", False) and not resolved_final) or has_outstanding(db, inbox, row.id), final_action_accepted=final_action_accepted)
        if category not in {"explicit", "confirm", "final_action"}:
            return outcome
        # One proposal per source. Publication, not worker latency, starts the five seconds.
        if not data.get("end_intent"):
            data["end_intent"] = {"id": uid(), "source": source["marker"], "category": category,
                                  "work_id": row.id}
            inbox.content_ciphertext = seal(data)
        return {"status": "end_proposed", "voice_ended": False}


def candidate_valid(db, inbox, data, candidate):
    if (inbox.closed or inbox.expires_at <= now() or not candidate
            or data.get("user_marker") != candidate.get("source")):
        return False
    if candidate["category"] == "explicit":
        return True
    if has_outstanding(db, inbox, candidate.get("work_id")):
        return False
    if candidate["category"] == "final_action":
        work = db.get(AgentWork, candidate.get("work_id"))
        job = db.get(Job, candidate.get("work_id"))
        return bool(work and job and job.status == "succeeded" and not work.cancel_requested
                    and not work.result.get("errors") and not work.result.get("pending_confirmation")
                    and work.result.get("actions"))
    return candidate["category"] == "confirm"
