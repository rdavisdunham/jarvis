"""Migration numbers Eridani reserves for runs, so two concurrent runs on one repository never pick the same one.

A counter belongs to a host's execution target (the repository), not to an ExecutionTarget row: several rows may
name the same host target. Only targets that say where migrations live take part: the folder (and file pattern) set
in Settings on the counter, else the host's config (`migrations {dir, pattern?}` in hello). The host reports the
chain head only for its own folder, so a folder set in Settings that differs relies on the override. The next number is max(the host's reported chain head, the operator's override, the highest
held reservation) + 1, taken under the counter's row lock at plan freeze.

A reservation follows the task: a later run of the same task reuses it. It is released when the run ends without
finishing (stopped, failed, capped) or a later plan revision drops the migration, and then only when no higher
number was reserved since; otherwise the gap stays. A reservation cannot see migrations merged after it was made:
the host's merge-first prompt and tests/test_release_migrations.py cover that, not this counter.
"""

import re
from pathlib import PurePosixPath

from sqlalchemy import func, select

from .domain import DomainError, advisory, owned
from .models import now
from .session_models import AgentHost, ExecutionTarget, MigrationCounter, MigrationReservation

RELEASE_ON = ("stopped", "failed", "capped")


def host_config(host, host_target):
    """The host's `migrations` for a target as hello listed it, or None when the target has none."""
    for item in (host.caps or {}).get("targets") or [] if host else []:
        if isinstance(item, dict) and item.get("id") == host_target:
            spec = item.get("migrations")
            if isinstance(spec, dict) and isinstance(spec.get("dir"), str) and spec["dir"]:
                return {"dir": spec["dir"][:200], **({"pattern": str(spec["pattern"])[:200]} if spec.get("pattern") else {})}
    return None


def config(host, counter, host_target):
    """Where the target's migrations live: Settings (the counter) first, else the host; None when neither says."""
    if counter is not None and counter.dir:
        return {"dir": counter.dir, **({"pattern": counter.pattern} if counter.pattern else {}), "source": "eridani"}
    spec = host_config(host, host_target)
    return {**spec, "source": "host"} if spec else None


def reported_head(host, host_target, dir=None):
    """The newest migration number the host found on the fresh base (worktrees.report heads), or None. With `dir`,
    only a head the host read from that same folder counts."""
    heads = ((host.caps or {}).get("worktrees") or {}).get("heads") or [] if host else []
    for item in heads:
        if isinstance(item, dict) and item.get("target") == host_target and isinstance(item.get("head"), int) \
                and (dir is None or str(item.get("dir") or "").rstrip("/") == dir.rstrip("/")):
            return item["head"]
    return None


def counter_for(db, owner, host_id, host_target, *, create=True):
    """The counter row, locked; created under an advisory lock so two first reservations make one row."""
    query = select(MigrationCounter).where(MigrationCounter.owner_id == owner, MigrationCounter.host_id == host_id,
                                           MigrationCounter.host_target == host_target)
    row = db.scalar(query.with_for_update())
    if row is None and create:
        advisory(db, f"migration-counter:{owner}:{host_id}:{host_target}")
        row = db.scalar(query.with_for_update())
        if row is None:
            row = MigrationCounter(owner_id=owner, host_id=host_id, host_target=host_target)
            db.add(row)
            db.flush()
    return row


def highest_held(db, counter_id):
    return db.scalar(select(func.max(MigrationReservation.number)).where(
        MigrationReservation.counter_id == counter_id, MigrationReservation.status == "held"))


def target_of(db, run):
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    host = db.get(AgentHost, target.host_id) if target is not None and target.host_id else None
    return target, host


def plan_value(row, spec):
    return {"number": row.number, "dir": spec["dir"], "reservation_id": row.id}


def held_for_task(db, run):
    """The task's reservation still held (made by an earlier run of it), on this run's repository."""
    target, host = target_of(db, run)
    if target is None or host is None:
        return None, None
    counter = counter_for(db, run.owner_id, host.id, target.host_target, create=False)
    if counter is None:
        return None, None
    row = db.scalar(select(MigrationReservation).where(
        MigrationReservation.counter_id == counter.id, MigrationReservation.task_id == run.task_id,
        MigrationReservation.status == "held").order_by(MigrationReservation.created_at.desc()).limit(1)
        .with_for_update())
    return row, config(host, counter, target.host_target)


def reserve(db, run):
    """At plan freeze for a plan that adds a migration: the task's held number, else the next one. None when the
    run's target has no migrations configured (nothing to reserve)."""
    target, host = target_of(db, run)
    if target is None or host is None:
        return None
    counter = counter_for(db, run.owner_id, host.id, target.host_target, create=False)
    spec = config(host, counter, target.host_target)
    if spec is None:
        return None
    counter = counter or counter_for(db, run.owner_id, host.id, target.host_target)
    existing, _ = held_for_task(db, run)
    if existing is not None:
        existing.run_id = run.id
        return plan_value(existing, spec)
    head = reported_head(host, target.host_target, spec["dir"])
    number = max(head or 0, counter.override or 0, highest_held(db, counter.id) or 0) + 1
    row = MigrationReservation(owner_id=run.owner_id, counter_id=counter.id, task_id=run.task_id, run_id=run.id,
                               number=number)
    db.add(row)
    counter.updated_at = now()
    db.flush()
    return plan_value(row, spec)


def carry(db, run):
    """A run frozen without an interview (agent_control.freeze) keeps the number an earlier run of its task holds."""
    row, spec = held_for_task(db, run)
    if row is None or spec is None:
        return None
    row.run_id = run.id
    return plan_value(row, spec)


def release(db, reservation_id):
    """Free a held number unless a higher one was reserved on the same counter since; then the gap stays."""
    row = db.get(MigrationReservation, reservation_id, with_for_update=True) if reservation_id else None
    if row is None or row.status != "held":
        return False
    db.get(MigrationCounter, row.counter_id, with_for_update=True)  # Serializes with reserve().
    if (highest_held(db, row.counter_id) or 0) > row.number:
        return False
    row.status, row.released_at = "released", now()
    db.flush()
    return True


def on_run_end(db, run):
    """A run that ends without finishing releases the number it holds (see release)."""
    if run.state not in RELEASE_ON:
        return
    for row in db.scalars(select(MigrationReservation).where(MigrationReservation.run_id == run.id,
                                                             MigrationReservation.status == "held")):
        release(db, row.id)


def render(migration):
    """The line render_plan adds for a plan with a reserved number."""
    number = f"{int(migration['number']):04d}"
    return (f"Migration: this plan's migration uses number {number} in {migration['dir']} (reserved by Eridani; no "
            f"other run gets it). Set down_revision to the current head of that chain; if main gained a migration "
            f"after you started, merge main, re-point down_revision and renumber.")


# ---------------------------------------------------------------- Settings


def listing(db, owner):
    """Every target the hosts advertise, with where its migrations live (Settings or the host), the host's head for
    that folder, the override, the last reservation and the next number. Unconfigured targets take no part."""
    out = []
    for host in db.scalars(select(AgentHost).where(AgentHost.owner_id == owner).order_by(AgentHost.created_at)):
        for item in (host.caps or {}).get("targets") or []:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            counter = counter_for(db, owner, host.id, item["id"], create=False)
            spec, host_spec = config(host, counter, item["id"]), host_config(host, item["id"])
            held = highest_held(db, counter.id) if counter else None
            head = reported_head(host, item["id"], spec["dir"]) if spec else None
            override = counter.override if counter else None
            out.append({"host_id": host.id, "host_name": host.name, "host_target": item["id"],
                        "configured": spec is not None, "source": spec["source"] if spec else None,
                        "dir": spec["dir"] if spec else None, "pattern": (spec or {}).get("pattern") or None,
                        "host_dir": host_spec["dir"] if host_spec else None,
                        "head": head, "override": override, "last_reserved": held,
                        "next": max(head or 0, override or 0, held or 0) + 1 if spec else None,
                        "reported_at": ((host.caps or {}).get("worktrees") or {}).get("at")})
    return out


def clean_dir(value):
    """A folder relative to the repository root: no absolute path, no `..`, one line."""
    value = (value or "").strip()
    if not value:
        return None
    if value.startswith(("/", "~")):
        raise DomainError("INVALID_ARGUMENT", "The migrations folder is a path relative to the repository, such as "
                                              "migrations/versions.")
    value = value.rstrip("/")
    if not value or len(value) > 200 or "\\" in value or re.search(r"[\x00-\x1f:]", value) \
            or ".." in PurePosixPath(value).parts:
        raise DomainError("INVALID_ARGUMENT", "The migrations folder is a path relative to the repository, such as "
                                              "migrations/versions.")
    return value


def clean_pattern(value):
    value = (value or "").strip()
    if not value:
        return None
    try:
        compiled = re.compile(value)
    except re.error:
        raise DomainError("INVALID_ARGUMENT", "The file pattern is not a valid regular expression.") from None
    if len(value) > 200 or compiled.groups < 1:
        raise DomainError("INVALID_ARGUMENT", "The file pattern needs a group around the number, such as ^(\\d{4})_.")
    return value


def configure(db, owner, host_id, host_target, *, dir=None, pattern=None, override=None):
    """Settings: the folder and file pattern (empty: use the host's), and the chain-head override (empty: none)."""
    host = owned(db, AgentHost, host_id, owner)
    if not any(isinstance(t, dict) and t.get("id") == host_target for t in (host.caps or {}).get("targets") or []):
        raise DomainError("NOT_FOUND", "That remotea server does not list this target.", 404)
    dir, pattern = clean_dir(dir), clean_pattern(pattern)
    if pattern and not dir:
        raise DomainError("INVALID_ARGUMENT", "Set the folder together with its file pattern.")
    counter = counter_for(db, owner, host.id, host_target)
    counter.dir, counter.pattern, counter.override, counter.updated_at = dir, pattern, override, now()
    db.flush()
    return {"items": listing(db, owner)}
