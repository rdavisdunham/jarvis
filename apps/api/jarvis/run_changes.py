"""What a run changed, as evidence for the judge.

The judge used to see the agent's reply (a claim), the verify result, CI and the pull request record, but never the
change itself. After every execution turn that will be judged, Eridani asks the run's host for the change with the
read-only `run.changes` command (sent only to a host whose hello caps list `run.changes`): the host reads the run's
worktree with git and answers with the changed files, the untracked files, and the patch against the point where the
branch left its base (lock files left out, secrets redacted, cut to size).

The answer is stored as an attempt of kind `changes` for the turn's (generation, turn_seq). The judge's evidence
takes the newest one for its own turn (`evidence`). The request and the judge can race when no verification runs in
between, so the judge job waits a short while for an answer that is still on its way (`wait`); it never waits for a
host that is not asked, and it judges without the change when none arrives.
"""
import time
from datetime import timedelta

from sqlalchemy import select

from . import run_coordinator as coordinator
from .db import session_scope
from .models import now
from .session_models import AgentAttempt, AgentCommand, AgentRun, ExecutionTarget

KIND = "changes"
COMMAND = "run.changes"
EXPIRY = timedelta(minutes=2)
# What the host is asked for (and what is stored), and what the judge is shown of it.
HOST_PATCH_CHARS = 60_000
JUDGE_PATCH_CHARS = 30_000
JUDGE_FILES = 100
# How long the judge job waits for an answer that is on its way.
WAIT = 20.0
SETTLED = ("executed", "failed", "rejected", "uncertain", "expired", "canceled", "superseded")


def offered(db, run):
    return bool(run.host_id and run.branch and coordinator.host_cap(db, run.host_id, COMMAND))


def request(db, run):
    """Ask the host what the run changed as of this turn. None when the host does not offer it."""
    if not offered(db, run):
        return None
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    body = {"branch": run.branch, "turn_seq": run.turn_seq, "max_chars": HOST_PATCH_CHARS,
            **({"base": run.base_branch} if run.base_branch else {}),
            **({"target": target.host_target} if target else {})}
    return coordinator.command(db, run, COMMAND, body, expires_at=now() + EXPIRY, source="eridani")


def on_outcome(db, run, row, state, detail="", result=None):
    """The host's answer (run_coordinator.on_outcome): recorded for the turn it was asked for; the run is not moved."""
    payload = row.payload or {}
    ok = state == "executed" and isinstance(result, dict)
    db.add(AgentAttempt(owner_id=run.owner_id, run_id=run.id, generation=row.generation, kind=KIND,
                        turn_seq=payload.get("turn_seq", run.turn_seq), plan_revision=run.plan_revision,
                        status="done" if ok else "failed",
                        result=clean(result) if ok else {"detail": (detail or state)[:300]}))


def clean(result):
    """The host's result, bounded: only the fields the judge reads, in the sizes it was asked for."""
    def paths(value, limit):
        return [str(p)[:400] for p in value[:limit]] if isinstance(value, list) else []

    files = [{"path": str(f.get("path", ""))[:400], "added": f.get("added"), "removed": f.get("removed")}
             for f in (result.get("files") or [])[:300] if isinstance(f, dict)]
    patch = str(result.get("patch") or "")[:HOST_PATCH_CHARS]
    return {"head": str(result.get("head") or "")[:64], "base_ref": str(result.get("base_ref") or "")[:200],
            "base_sha": str(result.get("base_sha") or "")[:64], "files": files,
            "files_total": int(result.get("files_total") or len(files)),
            "untracked": paths(result.get("untracked"), 100),
            "untracked_total": int(result.get("untracked_total") or 0),
            "uncommitted": bool(result.get("uncommitted")), "patch": patch,
            "patch_truncated": bool(result.get("patch_truncated")) or len(patch) < int(result.get("patch_chars") or 0)}


def latest(db, run_id, generation, turn_seq):
    return db.scalars(select(AgentAttempt).where(
        AgentAttempt.run_id == run_id, AgentAttempt.kind == KIND, AgentAttempt.generation == generation,
        AgentAttempt.turn_seq == turn_seq).order_by(AgentAttempt.created_at.desc()).limit(1)).first()


def evidence(db, run, attempt):
    """The judge's `changes`: the change as of the judged turn, or why there is none to show."""
    row = latest(db, run.id, attempt.generation, attempt.turn_seq)
    if row is None:
        why = "the host was asked but has not answered" if offered(db, run) else "this host does not report changes"
        return {"available": False, "reason": why}
    if row.status != "done":
        return {"available": False, "reason": (row.result or {}).get("detail") or "the host could not read the changes"}
    r = row.result or {}
    patch = r.get("patch") or ""
    return {"available": True, "head": r.get("head"), "base": r.get("base_ref"),
            "files": (r.get("files") or [])[:JUDGE_FILES], "files_total": r.get("files_total", 0),
            "untracked": r.get("untracked") or [], "uncommitted": bool(r.get("uncommitted")),
            "patch": patch[:JUDGE_PATCH_CHARS],
            "patch_truncated": bool(r.get("patch_truncated")) or len(patch) > JUDGE_PATCH_CHARS}


def pending(db, run_id, generation, turn_seq):
    """A request for this turn is on its way and has no answer yet."""
    if latest(db, run_id, generation, turn_seq) is not None:
        return False
    rows = db.scalars(select(AgentCommand).where(
        AgentCommand.run_id == run_id, AgentCommand.kind == COMMAND, AgentCommand.generation == generation,
        AgentCommand.status.notin_(SETTLED), AgentCommand.expires_at > now()))
    return any((row.payload or {}).get("turn_seq") == turn_seq for row in rows)


def wait(attempt_id, timeout=WAIT, step=1.0, sleep=time.sleep):
    """Before a judge attempt reads its evidence: give an answer that is on its way up to `timeout` seconds."""
    waited = 0.0
    while waited < timeout:
        with session_scope() as db:
            attempt = db.get(AgentAttempt, attempt_id)
            run = db.get(AgentRun, attempt.run_id) if attempt else None
            if run is None or not pending(db, run.id, attempt.generation, attempt.turn_seq):
                return waited
        sleep(step)
        waited += step
    return waited
