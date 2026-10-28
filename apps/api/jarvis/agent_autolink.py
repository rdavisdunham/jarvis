"""The dream sequence's session auto-link (decision 14, C12): file unfiled agent sessions under the task
they were evidently about, once a day, with a durable Undo.

Eligible sessions are unfiled or filed only by a folder rule (whose home may be a project with its own
task: the dream refines that to the issue underneath). Candidates come from the session's record chain:
tasks of the records under the record it is filed in, plus tasks named by exact Linear identifiers
resolved in the owner's connected workspace. Open tasks and tasks completed in the last RECENT_DAYS count; historical sessions
often describe work that has since closed.

Ranking is a heuristic, not a calibrated confidence: the cosine between the session's embedding and the
task record's search vectors, plus ISSUE_BONUS when the session names the task's issue. A session is
linked only when the top score reaches THRESHOLD and leads the runner-up by MARGIN. The constants are
provisional (tuned on the labelled fixtures in tests/test_agent_autolink.py, not on real sessions yet).

An automatic link is `binding_source="auto"` with the evidence revision it was made at; resolve() keeps it
through collector refreshes and rule rebinds until that evidence changes. Undo (or clearing the link)
writes an `autolink_rejections` tombstone for (session, task) that blocks the pair for good: a rejected
task never wins again, and if it would rank first the session stays unlinked. Inferred links never
authorize starting work: runs start only from a task, with the execution-target rules.
"""

from datetime import timedelta

from sqlalchemy import or_, select

from . import agent_sessions as sessions
from .db import session_scope
from .domain import DomainError, emit, enqueue_job, owned
from .models import Job, LinearConnection, OwnerSettings, StructureRecord, Task, now
from .search_models import SearchDocument
from .session_models import AgentSession, AutolinkRejection

THRESHOLD = 0.62  # provisional
MARGIN = 0.12  # provisional
ISSUE_BONUS = 0.25  # provisional
RECENT_DAYS = 30
INTERVAL = timedelta(hours=20)
WINDOW = timedelta(days=30)
BATCH = 500


def evidence_revision(row):
    """What the link was inferred from. last_prompt and turn counts change every turn and are left out."""
    return sessions.digest([row.title, row.first_prompt, row.repo, row.branch, sessions.normalize_path(row.cwd),
                            sorted(r["identifier"] for r in row.issue_refs or []), row.summary_fingerprint])


# ---------------------------------------------------------------- ranking (pure)


def choose(scored, rejected=frozenset()):
    """scored: {task_id: score}. Returns (task_id, score, margin), or None when nothing is clear enough."""
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return None
    (task, top), runner = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
    if task in rejected or top < THRESHOLD or top - runner < MARGIN:
        return None
    return task, round(top, 4), round(top - runner, 4)


def score(session_vector, task_vectors, named):
    from .memory_learning import cosine

    best = max((cosine(session_vector, v) for v in task_vectors or []), default=0.0)
    return best + (ISSUE_BONUS if named else 0.0)


# ---------------------------------------------------------------- candidates


def workspace_issues(db, owner, refs):
    """Exact identifiers → task ids, only for issues of the currently connected Linear workspace."""
    from .models import LinearIssue

    conn = db.get(LinearConnection, owner)
    keys = [r["identifier"] for r in refs or [] if r.get("identifier")]
    if not keys or conn is None or not conn.enabled:
        return set()
    return set(db.scalars(select(LinearIssue.task_id).where(
        LinearIssue.owner_id == owner, LinearIssue.workspace_id == conn.workspace_id,
        LinearIssue.snapshot["identifier"].astext.in_(keys))))


def candidate_tasks(db, owner, row):
    from .record_contents import rows_under

    ids = set()
    if row.record_id:
        home = db.get(StructureRecord, row.record_id)
        if home is not None and home.owner_id == owner:  # Below the home, not the home's own task.
            ids |= {r.task_id for r in rows_under(db, owner, home.id) if r.task_id and not r.archived}
    named = workspace_issues(db, owner, row.issue_refs)
    ids |= named
    if not ids:
        return [], named
    cutoff = now() - timedelta(days=RECENT_DAYS)
    tasks = db.scalars(select(Task).where(
        Task.owner_id == owner, Task.id.in_(ids), Task.archived.is_(False), Task.is_template.is_(False),
        Task.status != "cancelled",
        or_(Task.status != "completed", Task.completed_at >= cutoff)))
    return list(tasks), named


def task_vectors(db, owner, task_ids, cache):
    from .memory_learning import EMBEDDING_MODEL

    missing = [t for t in task_ids if t not in cache]
    if missing:
        records = dict(db.execute(select(StructureRecord.task_id, StructureRecord.id).where(
            StructureRecord.owner_id == owner, StructureRecord.task_id.in_(missing))).all())
        documents = {d.target_key: d for d in db.scalars(select(SearchDocument).where(
            SearchDocument.owner_id == owner, SearchDocument.embedding_model == EMBEDDING_MODEL,
            SearchDocument.target_key.in_(["record:" + r for r in records.values()])))}
        for task in missing:
            document = documents.get("record:" + records.get(task, ""))
            cache[task] = document.vectors if document else []
    return {t: cache[t] for t in task_ids}


def eligible(db, owner):
    from .memory_learning import EMBEDDING_MODEL

    return db.scalars(select(AgentSession).where(
        AgentSession.owner_id == owner, AgentSession.archived.is_(False), AgentSession.run_id.is_(None), AgentSession.binding_source.in_(("", "path_rule")),
        AgentSession.embedding_model == EMBEDDING_MODEL,
        or_(AgentSession.last_active_at.is_(None), AgentSession.last_active_at >= now() - WINDOW))
        .order_by(AgentSession.last_active_at.desc().nulls_last()).limit(BATCH))


def link_session(db, row, task_id):
    row.task_id, row.record_id = task_id, sessions.record_for_task(db, row.owner_id, task_id)
    row.binding_source, row.binding_evidence, row.updated_at = "auto", evidence_revision(row), now()
    emit(db, row.owner_id, "session.changed", row.id)


def cycle(db, owner):
    """One dream pass for an owner. Returns counts for the job result."""
    rejected = {}
    for session_pk, task_id in db.execute(select(AutolinkRejection.session_pk, AutolinkRejection.task_id)
                                          .where(AutolinkRejection.owner_id == owner)):
        rejected.setdefault(session_pk, set()).add(task_id)
    cache, counts = {}, {"considered": 0, "linked": 0, "blocked_by_undo": 0}
    for row in list(eligible(db, owner)):
        tasks, named = candidate_tasks(db, owner, row)
        if not tasks or not row.embedding:
            continue
        counts["considered"] += 1
        vectors = task_vectors(db, owner, [t.id for t in tasks], cache)
        scored = {t.id: score(row.embedding, vectors[t.id], t.id in named) for t in tasks}
        picked = choose(scored)
        if picked is None:
            continue
        if picked[0] in rejected.get(row.id, set()):
            counts["blocked_by_undo"] += 1
            continue
        link_session(db, row, picked[0])
        counts["linked"] += 1
    return counts


# ---------------------------------------------------------------- the daily job


def owners(db):
    return [row.owner_id for row in db.scalars(select(OwnerSettings))
            if (row.values or {}).get("remotea_enabled")]


def queue_autolink(db, clock=None):
    """From supervisor_cycle: one job per enabled owner, at least INTERVAL apart."""
    clock = clock or now()
    queued = []
    for owner in owners(db):
        last = db.scalar(select(Job.created_at).where(Job.owner_id == owner, Job.kind == "agent_autolink")
                         .order_by(Job.created_at.desc()).limit(1))
        if last is not None and clock - last < INTERVAL:
            continue
        queued.append(enqueue_job(db, owner, "agent_autolink", {}))
    return queued


def process(job_id):
    with session_scope() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if not job or job.status in {"succeeded", "cancelled", "failed"}:
            return
        job.status = "running"
        counts = cycle(db, job.owner_id)
        job.status, job.finished_at, job.result = "succeeded", now(), counts


# ---------------------------------------------------------------- Undo


def reject(db, row):
    """Tombstone the (session, task) pair of an automatic link; nothing re-proposes it."""
    if not row.task_id:
        return
    exists = db.scalar(select(AutolinkRejection.id).where(AutolinkRejection.session_pk == row.id,
                                                          AutolinkRejection.task_id == row.task_id))
    if exists is None:
        db.add(AutolinkRejection(owner_id=row.owner_id, session_pk=row.id, task_id=row.task_id,
                                 evidence_revision=row.binding_evidence or evidence_revision(row)))


def undo(db, owner, session_pk):
    row = owned(db, AgentSession, session_pk, owner, lock=True)
    if row.binding_source != "auto":
        raise DomainError("NOT_AUTO", "Only a link the dream sequence made can be undone here.", 409)
    reject(db, row)
    row.task_id = row.record_id = None
    row.binding_source, row.binding_evidence, row.updated_at = "", "", now()
    sessions.resolve(db, row)  # A folder rule may still file it under its client or project.
    db.flush()
    emit(db, owner, "session.changed", row.id)
    return sessions.public(db, row)
