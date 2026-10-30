"""Migration numbers Eridani reserves for runs, so two concurrent runs on one repository never pick the same one.

A counter belongs to a host's execution target (the repository), not to an ExecutionTarget row: several rows may
name the same host target. Only targets whose host config says where migrations live (`migrations {dir, pattern?}`
in hello) take part. The next number is max(the host's reported chain head, the operator's override, the highest
held reservation) + 1, taken under the counter's row lock at plan freeze.

A reservation follows the task: a later run of the same task reuses it. It is released when the run ends without
finishing (stopped, failed, capped) or a later plan revision drops the migration, and then only when no higher
number was reserved since; otherwise the gap stays. A reservation cannot see migrations merged after it was made:
the host's merge-first prompt and tests/test_release_migrations.py cover that, not this counter.
"""

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


def reported_head(host, host_target):
    """The newest migration number the host found on the fresh base (worktrees.report heads), or None."""
    heads = ((host.caps or {}).get("worktrees") or {}).get("heads") or [] if host else []
    for item in heads:
        if isinstance(item, dict) and item.get("target") == host_target and isinstance(item.get("head"), int):
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
    return row, host_config(host, target.host_target)


def reserve(db, run):
    """At plan freeze for a plan that adds a migration: the task's held number, else the next one. None when the
    run's target has no migrations configured (nothing to reserve)."""
    target, host = target_of(db, run)
    spec = host_config(host, target.host_target) if target is not None else None
    if spec is None:
        return None
    counter = counter_for(db, run.owner_id, host.id, target.host_target)
    existing, _ = held_for_task(db, run)
    if existing is not None:
        existing.run_id = run.id
        return plan_value(existing, spec)
    number = max(reported_head(host, target.host_target) or 0, counter.override or 0, highest_held(db, counter.id) or 0) + 1
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
    """Every advertised target with migrations configured: the host's head, the override, the last reservation."""
    out = []
    for host in db.scalars(select(AgentHost).where(AgentHost.owner_id == owner).order_by(AgentHost.created_at)):
        for item in (host.caps or {}).get("targets") or []:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            spec = host_config(host, item["id"])
            if spec is None:
                continue
            counter = counter_for(db, owner, host.id, item["id"], create=False)
            held = highest_held(db, counter.id) if counter else None
            head, override = reported_head(host, item["id"]), counter.override if counter else None
            out.append({"host_id": host.id, "host_name": host.name, "host_target": item["id"], "dir": spec["dir"],
                        "head": head, "override": override, "last_reserved": held,
                        "next": max(head or 0, override or 0, held or 0) + 1,
                        "reported_at": ((host.caps or {}).get("worktrees") or {}).get("at")})
    return out


def set_override(db, owner, host_id, host_target, override):
    host = owned(db, AgentHost, host_id, owner)
    if host_config(host, host_target) is None:
        raise DomainError("INVALID_ARGUMENT", "That target has no migrations folder configured on the host.")
    counter = counter_for(db, owner, host.id, host_target)
    counter.override, counter.updated_at = override, now()
    db.flush()
    return {"items": listing(db, owner)}

