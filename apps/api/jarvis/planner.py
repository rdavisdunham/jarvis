"""Deterministic single-person planning and an atomic, revalidated local save."""

import hashlib
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from uuid import UUID

from sqlalchemy import delete

from .db import session_scope
from .domain import DomainError, advisory, check_revision, owned, zone
from .google_auth import seal, unseal
from .models import Command, Task, now, uid
from .planner_schema import PlanRequest

MAX_NODES = 120000


def instant(value):
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None:
            raise ValueError()
        return result.astimezone(UTC)
    except (ValueError, TypeError):
        raise DomainError("INVALID_ARGUMENT", "Use a complete date/time with an explicit UTC offset.") from None


def constraints(request):
    begin, end = instant(request.start), instant(request.end)
    zone(request.timezone)
    if not timedelta(0) < end - begin <= timedelta(days=7):
        raise DomainError("INVALID_ARGUMENT", "Plan within a positive window of at most seven days.")
    ids = {item.task_id for item in request.tasks}
    if len(ids) != len(request.tasks):
        raise DomainError("INVALID_ARGUMENT", "Include each task once in the plan.")
    result = {}
    for item in request.tasks:
        if item.task_id in item.after_task_ids or not set(item.after_task_ids) <= ids:
            raise DomainError("INVALID_ARGUMENT", "Dependencies must name other tasks in this plan.")
        result[item.task_id] = {
            "minutes": item.minutes,
            "start": max(begin, instant(item.not_before) if item.not_before else begin),
            "end": min(end, instant(item.not_after) if item.not_after else end),
            "after": set(item.after_task_ids),
        }
    visited = set()
    while len(visited) < len(ids):
        ready = {key for key, value in result.items() if value["after"] <= visited} - visited
        if not ready:
            raise DomainError("INVALID_ARGUMENT", "These planning dependencies form a cycle.")
        visited |= ready
    return begin, end, result


def read_availability(owner, request):
    from .google_calendar import availability
    from .planning import local_availability

    begin, end, _ = constraints(request)
    if request.scope == "local_only":
        data = local_availability(owner, begin, end, 1)
    else:
        data = availability(owner, request.start, request.end, 5)
        if data.get("source") == "eridani_only" or data.get("calendar_count", 0) < 1:
            return {"status": "unavailable", "reason": "Selected Google calendars were not verified. "
                    "Connect them, or explicitly request planning against Eridani only."}
    if data.get("status") != "fresh":
        return {"status": "unavailable", "reason": data.get("reason", "Availability could not be confirmed.")}
    # Recompute every free interval from busy instants. A minimum free-slot filter
    # must not erase gaps that could hold one of the requested shorter tasks.
    busy = []
    for block in data.get("busy", []):
        a, b = max(begin, instant(block["start"])), min(end, instant(block["end"]))
        if b > a:
            busy.append((a, b))
    merged = []
    for a, b in sorted(busy):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    free, cursor = [], begin
    for a, b in merged + [(end, end)]:
        if a > cursor:
            free.append((cursor, a))
        cursor = max(cursor, b)
    return {**data, "free_intervals": free, "busy_intervals": merged}


def solve(request, free):
    """Enumerate chronological task orders, using earliest placement per order.

    For this non-preemptive single-person model with release times, deadlines,
    precedences and fixed busy intervals, shifting an earliest task later cannot
    improve the final finish. Enumerating all eligible orders therefore proves the
    minimum final finish when search completes. No multi-resource claim is made.
    """
    begin, end, requirements = constraints(request)
    best, best_end, nodes = None, None, 0
    exhausted = False

    def visit(placed, cursor):
        nonlocal best, best_end, nodes, exhausted
        nodes += 1
        if nodes > MAX_NODES:
            exhausted = True
            return
        if best_end is not None and cursor >= best_end:
            return
        if len(placed) == len(requirements):
            best, best_end = dict(placed), cursor
            return
        for key, rule in sorted(requirements.items(), key=lambda pair: (pair[1]["end"], pair[0])):
            if key in placed or not rule["after"] <= placed.keys():
                continue
            earliest = max(cursor, rule["start"])
            duration = timedelta(minutes=rule["minutes"])
            for a, b in free:
                start = max(a, earliest)
                finish = start + duration
                if finish <= min(b, rule["end"], end):
                    placed[key] = (start, finish)
                    visit(placed, finish)
                    placed.pop(key)
                    break

    visit({}, begin)
    return best, not exhausted, nodes


def checked_tasks(db, owner, request):
    rows = {}
    for item in request.tasks:
        row = owned(db, Task, item.task_id, owner)
        check_revision(row, item.expected_revision)
        if row.archived or row.is_template or row.status in {"completed", "cancelled"}:
            raise DomainError("INVALID_ARGUMENT", "Plan active tasks, not archived work or routine templates.")
        rows[row.id] = row
    return rows


def propose(owner, arguments):
    request = PlanRequest.model_validate(arguments)
    constraints(request)
    with session_scope() as db:
        titles = {key: row.title for key, row in checked_tasks(db, owner, request).items()}
    available = read_availability(owner, request)
    if available["status"] != "fresh":
        return {**available, "blocks": [], "saved": False}
    placed, proven, nodes = solve(request, available["free_intervals"])
    if not placed:
        return {
            "status": "infeasible" if proven else "search_limit", "saved": False, "blocks": [],
            "reason": "The required durations, time windows and dependencies do not fit the confirmed free time."
            if proven else "No plan was found within the search limit; infeasibility is not proven.",
            "required_minutes": sum(t.minutes for t in request.tasks),
            "available_minutes": sum(int((b-a).total_seconds() / 60) for a, b in available["free_intervals"]),
            "source": available["source"], "timezone": request.timezone,
            "busy": [{"start": a.astimezone(zone(request.timezone)).isoformat(),
                      "end": b.astimezone(zone(request.timezone)).isoformat()}
                     for a, b in available["busy_intervals"]],
            "search_nodes": nodes,
        }
    local = zone(request.timezone)
    blocks = [{"task_id": key, "title": titles[key], "start": a.astimezone(local).isoformat(),
               "end": b.astimezone(local).isoformat()} for key, (a, b) in
              sorted(placed.items(), key=lambda item: item[1][0])]
    expires = now() + timedelta(minutes=15)
    plan_id = uid()
    sealed = seal({"purpose": "planning_proposal", "owner": owner, "plan_id": plan_id,
                   "expires_at": expires.isoformat(), "request": request.model_dump(), "blocks": blocks})
    # Keep the large encrypted payload server-side. Models copy one short ID.
    # Uncommitted proposals expire; committed plan receipts remain replayable.
    with session_scope() as db:
        db.execute(delete(Command).where(
            Command.owner_id == owner, Command.id.startswith("planning-proposal:"),
            Command.created_at < now() - timedelta(minutes=15)))
        db.add(Command(owner_id=owner, id="planning-proposal:" + plan_id,
                       request_hash=hashlib.sha256(sealed.encode()).hexdigest(),
                       result={"command_id": "planning-proposal:" + plan_id, "status": "succeeded",
                               "data": {"sealed_proposal": sealed}, "committed_at": now().isoformat()}))
    return {
        "status": "ready", "saved": False, "blocks": blocks, "timezone": request.timezone,
        "scope": request.scope, "source": available["source"], "checked_at": available.get("checked_at"),
        "optimality": "proven_earliest_finish" if proven else "feasible_not_proven_optimal",
        "finish": blocks[-1]["end"], "search_nodes": nodes, "plan_token": plan_id,
        "expires_at": expires.isoformat(),
        "next": "The proposal is not saved. Commit only if the owner requested scheduling; otherwise show it.",
    }


def commit(db, owner, token):
    from .planning import mutate
    from .planning_schema import PlanningCreate

    try:
        if str(UUID(token)) != token:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise DomainError("PLAN_INVALID", "Create a new verified plan for this owner.") from None
    receipt_id = "planning-plan:" + token
    advisory(db, f"workspace:{owner}")
    advisory(db, f"plan:{owner}:{receipt_id}")
    previous = db.get(Command, (owner, receipt_id))
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    if previous:
        if previous.request_hash != token_hash:
            raise DomainError("PLAN_INVALID", "The plan receipt does not match.")
        return previous.result["data"]
    stored = db.get(Command, (owner, "planning-proposal:" + token))
    try:
        if stored is None:
            raise ValueError()
        sealed = stored.result["data"]["sealed_proposal"]
        if stored.request_hash != hashlib.sha256(sealed.encode()).hexdigest():
            raise ValueError()
        proposal = unseal(sealed)
        if (proposal.get("purpose") != "planning_proposal" or proposal.get("owner") != owner
                or proposal.get("plan_id") != token):
            raise ValueError()
        request = PlanRequest.model_validate(proposal["request"])
    except (ValueError, KeyError, TypeError, DomainError):
        raise DomainError("PLAN_INVALID", "This plan is missing or expired. Generate a new verified proposal.") from None
    if instant(proposal["expires_at"]) <= now():
        raise DomainError("PLAN_EXPIRED", "Availability may have changed. Generate a new proposal.", 409)
    rows = checked_tasks(db, owner, request)
    availability = read_availability(owner, request)
    if availability["status"] != "fresh":
        raise DomainError("AVAILABILITY_UNKNOWN", availability["reason"], 409)
    _, _, rules = constraints(request)
    intervals = {}
    for block in proposal["blocks"]:
        a, b = instant(block["start"]), instant(block["end"])
        key = block["task_id"]
        rule = rules[key]
        if not any(start <= a < b <= end for start, end in availability["free_intervals"]):
            raise DomainError("PLAN_CONFLICT", "A proposed block is no longer free. Generate a new plan.", 409)
        if b-a != timedelta(minutes=rule["minutes"]) or a < rule["start"] or b > rule["end"]:
            raise DomainError("PLAN_INVALID", "The block does not satisfy its time constraints.")
        intervals[key] = (a, b)
    if set(intervals) != set(rules):
        raise DomainError("PLAN_INVALID", "The plan must contain every requested task exactly once.")
    ordered = sorted(intervals.values())
    if any(b > c for (_, b), (c, _) in pairwise(ordered)):
        raise DomainError("PLAN_INVALID", "Proposed work blocks overlap.")
    for key, rule in rules.items():
        if any(intervals[parent][1] > intervals[key][0] for parent in rule["after"]):
            raise DomainError("PLAN_INVALID", "A dependency is not finished before its next task.")
    saved = []
    for block in proposal["blocks"]:
        saved.append(mutate(db, owner, "planning.create", PlanningCreate(
            kind="block", task_id=block["task_id"], title=rows[block["task_id"]].title,
            start=block["start"], end=block["end"], timezone=request.timezone)))
    result = {"plan_id": proposal["plan_id"], "entries": saved, "saved_count": len(saved),
              "scope": request.scope, "source": availability["source"], "google_published": False,
              "note": "Local blocks saved; task deadlines, planned dates and alerts are unchanged."}
    # Token-level idempotency survives new client command IDs and process restarts.
    # This receipt and all blocks commit in the same database transaction.
    db.add(Command(owner_id=owner, id=receipt_id, request_hash=token_hash,
                   result={"command_id": receipt_id, "status": "succeeded", "data": result, "committed_at": now().isoformat()}))
    return result
