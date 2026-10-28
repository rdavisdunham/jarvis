"""A task's execution plan: ordered steps with stable ids, success criteria, verify config and turn cap.

The plan on a task is a template; agent runs (stage C) freeze their own revision. Every entrance
(browser, Eri, external bots) goes through task.update, so this module is the single normalization
path. Plan fields never sync to Linear (linear_sync maps only core task fields).
"""

import secrets

from .domain import Criterion, DomainError, PlanStep, owned
from .models import Note, Task, now

PLAN_FIELDS = ("steps", "prd_note_id", "success_criteria", "verify", "turn_cap", "execution_target_id")


def new_id(prefix):
    return prefix + secrets.token_hex(4)


def actor():
    from .bot_access import current_id

    credential = current_id()
    return "bot:" + credential if credential else "user"


def stable(items, prefix, label):
    seen = set()
    for item in items:
        item["id"] = item.get("id") or new_id(prefix)
        if item["id"] in seen:
            raise DomainError("INVALID_ARGUMENT", f"Each {label} needs its own id.")
        seen.add(item["id"])
    return items


def normalize(db, owner, task, changes):
    """Validate plan fields in place; advance plan_revision when the plan actually changes."""
    if not any(key in changes for key in PLAN_FIELDS):
        return changes
    if "steps" in changes:
        previous = {s.get("id"): s for s in task.steps or []}
        steps = []
        for step in stable([PlanStep.model_validate(s).model_dump() for s in changes["steps"]], "s_", "step"):
            if step.get("task_id"):
                linked = owned(db, Task, step["task_id"], owner)
                if linked.id == task.id:
                    raise DomainError("INVALID_ARGUMENT", "A step cannot link to its own task.")
            old = previous.get(step["id"])
            core = {k: step.get(k) for k in ("title", "status", "note", "task_id")}
            if old and all(old.get(k) == v for k, v in core.items()):
                steps.append(old)
            else:
                steps.append({**core, "id": step["id"], "by": actor(), "updated_at": now().isoformat(),
                              "revision": (old or {}).get("revision", 0) + 1})
        changes["steps"] = steps
    if "success_criteria" in changes:
        criteria = stable([Criterion.model_validate(c).model_dump() for c in changes["success_criteria"]], "c_", "criterion")
        for criterion in criteria:
            if criterion["kind"] == "verify" and not criterion.get("verify_id"):
                raise DomainError("INVALID_ARGUMENT", "A verify criterion names the verify_id it needs to pass.")
        changes["success_criteria"] = criteria
    if changes.get("prd_note_id"):
        note = owned(db, Note, changes["prd_note_id"], owner)
        if note.archived:
            raise DomainError("NOT_FOUND", "That PRD note is archived.", 404)
    if any(key in changes and changes[key] != getattr(task, key) for key in PLAN_FIELDS):
        changes["plan_revision"] = (task.plan_revision or 0) + 1
    return changes
