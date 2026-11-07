"""Set up with Dani: the owner confirms what Dani should always keep in mind (docs/SETUP_WITH_DANI_PLAN.md, PR 6).

The synthesis stores core memory candidates on the setup session. `setup.memories_save` saves the ticked ones, as
edited, in one command:
- add: a new owner-confirmed fact flagged `core`, unless a live fact already says the same thing (the Luna extractor
  may have saved it from the same turns), which is flagged instead;
- keep: the existing fact is flagged `core` (edited wording supersedes it, like Correct);
- update: the corrected fact supersedes the existing one and is flagged `core`;
- retire: the existing fact stays saved but is no longer `core`.
Unticked candidates are discarded, never rejected. Each confirmed fact gets its own explicit Source, so deleting the
setup transcript later does not drop it. The About you summary is rebuilt afterwards.
"""

import hashlib

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .domain import Args, DomainError, emit, enqueue_job, owned
from .models import Memory, Source
from .setup import ACTIVE, personal, public
from .setup_models import SetupSession


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=36, description="A memory candidate ID from the setup state.")
    content: str = Field(min_length=1, max_length=2000, description="The fact as the owner confirmed or edited it.")
    include: bool = True


class MemoriesSave(Args):
    choices: list[Choice] = Field(max_length=40)


class Pin(Args):
    memory_id: str


COMMANDS = {"setup.memories_save": MemoriesSave}
MEMORY_COMMANDS = {"memory.pin": Pin, "memory.unpin": Pin}


def fold(text):
    return " ".join((text or "").split()).casefold()


def prints(text):
    """Both fingerprints in use: memory_learning's (extracted facts) and memory.capture's."""
    from .memory_learning import fingerprint

    return {fingerprint(text), hashlib.sha256(fold(text).encode()).hexdigest()}


def live(db, owner, memory_id):
    if not memory_id:
        return None
    row = db.get(Memory, memory_id, with_for_update=True)
    if not row or row.owner_id != owner or row.suppressed or row.merged_into_id or not row.content:
        return None
    source = db.get(Source, row.source_id)
    return row if source and source.deleted_at is None else None


def same_fact(db, owner, content):
    """A live fact with the same wording, if one exists (dedupe against facts learned from the same turns)."""
    for row in db.scalars(select(Memory).where(Memory.owner_id == owner, Memory.suppressed.is_(False),
                                               Memory.merged_into_id.is_(None),
                                               Memory.fingerprint.in_(prints(content)))):
        if live(db, owner, row.id):
            return row
    return None


def create(db, owner, content, candidate, command_id, supersedes=None):
    from .domain import capture_source
    from .memory_learning import fingerprint
    from .memory_review import attribution

    source = capture_source(db, owner, content, f"setup-memory:{command_id}:{candidate['id']}", explicit=True)
    row = Memory(owner_id=owner, source_id=source.id, content=content, attribution=attribution(command_id),
                 evidence=candidate.get("evidence") or "", fact_key=candidate.get("fact_key") or "",
                 fingerprint=fingerprint(content), core=True,
                 supersedes_id=supersedes.id if supersedes else None,
                 revision=(supersedes.revision + 1) if supersedes else 1)
    if supersedes:
        row.fact_key = supersedes.fact_key or row.fact_key
        supersedes.suppressed, supersedes.embedding, supersedes.core = True, None, False
    db.add(row)
    db.flush()
    enqueue_job(db, owner, "embed_memory", {"memory_id": row.id})
    return row


def save(db, owner, args, command_id):
    personal(db, owner)
    session = db.get(SetupSession, owner, with_for_update=True)
    if not session or session.status not in ACTIVE:
        raise DomainError("SETUP_NOT_ACTIVE", "Start setup with Dani first.", 409)
    candidates = {c["id"]: c for c in session.memory_candidates or []}
    if any(choice.id not in candidates for choice in args.choices):
        raise DomainError("REVISION_CONFLICT", "These suggestions changed. Review them again.", 409)
    confirmed, retired = [], []
    for choice in args.choices:
        if not choice.include:
            continue  # Discarded, not rejected: automatic learning may still save it later.
        candidate = candidates[choice.id]
        action, content = candidate["action"], choice.content
        existing = live(db, owner, candidate.get("existing_memory_id"))
        if action == "retire":
            if existing:
                existing.core = False
                retired.append(existing.id)
            continue
        if action in ("keep", "update") and existing is None:
            action = "add"
        if action == "keep" and fold(content) != fold(existing.content):
            action = "update"
        if action == "keep":
            existing.core = True
            row = existing
        elif action == "update":
            row = create(db, owner, content, candidate, command_id, supersedes=existing)
        else:
            row = same_fact(db, owner, content)
            if row is not None:
                row.core = True
            else:
                row = create(db, owner, content, candidate, command_id)
        confirmed.append(row.id)
    session.confirmed_memory_ids = list(dict.fromkeys([*(session.confirmed_memory_ids or []), *confirmed]))
    session.memory_candidates = []
    session.notes = {**(session.notes or {}), "memories_saved": True}
    session.revision += 1
    rebuild(db, owner)
    emit(db, owner, "memory.changed", owner)
    emit(db, owner, "setup.changed", owner, session.revision)
    return {**public(db, owner, session), "saved_ids": confirmed, "retired_ids": retired}


def rebuild(db, owner):
    """Queue an About you rebuild now rather than at the next daily check."""
    from .about_you import allowed, enabled, schedule
    from .domain import preferences

    if enabled() and allowed(preferences(db, owner)):
        schedule(db, owner)


def pin(db, owner, tool, args):
    row = owned(db, Memory, args.memory_id, owner, lock=True)
    if live(db, owner, row.id) is None:
        raise DomainError("NOT_FOUND", "That fact is no longer saved.", 404)
    row.core = tool == "memory.pin"
    rebuild(db, owner)
    emit(db, owner, "memory.changed", row.id)
    return {"id": row.id, "core": row.core, "content": row.content}


def mutate(db, owner, tool, args, command_id):
    if tool == "setup.memories_save":
        return save(db, owner, args, command_id)
    return pin(db, owner, tool, args)
