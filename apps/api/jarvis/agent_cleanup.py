"""Retire ended runs or a successfully replaced planner, using exact host facts."""
import json
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select

from .session_models import TERMINAL_RUN_STATES, AgentCommand, AgentHost, AgentRun, AgentSession


def binding(value):
    if not isinstance(value, dict) or not isinstance(value.get("surf"), str) or not value["surf"]:
        return None
    if any(type(value.get(key)) is not int or value[key] < 1 for key in ("incarnation", "generation")):
        return None
    return {key: value[key] for key in ("surf", "incarnation", "generation")}


def confirm_handoff(run, surf, incarnation, generation):
    """SessionStart, a root turn or a live provider snapshot proves the replacement attached."""
    handoff = (run.sync_state or {}).get("handoff") or {}
    worker = binding({"surf": surf, "incarnation": incarnation, "generation": generation})
    if not (handoff.get("done") and run.session_pk and worker and surf == run.surf == handoff.get("surf")
            and surf != handoff.get("from_surf") and generation == run.generation):
        return
    if handoff.get("attached_binding") != worker:
        run.sync_state = {**(run.sync_state or {}), "handoff": {**handoff, "attached_binding": worker}}


def retired_body(run):
    """The only active-run cleanup allowed is the exact, proven planner replacement."""
    if run.state in TERMINAL_RUN_STATES or run.state == "stopping":
        return None
    handoff = (run.sync_state or {}).get("handoff") or {}
    known = (run.sync_state or {}).get("cleanup_surfaces") or {}
    planner = binding(known.get(handoff.get("from_surf")) or handoff.get("from_binding"))
    worker = binding(handoff.get("attached_binding"))
    if not (handoff.get("done") and handoff.get("from_session_pk") and run.session_pk
            and handoff["from_session_pk"] != run.session_pk and planner and worker
            and worker["surf"] == run.surf == handoff.get("surf") and planner["surf"] == handoff.get("from_surf")
            and planner["surf"] != worker["surf"] and planner["generation"] <= run.generation
            and worker["generation"] == run.generation):
        return None
    current = known.get(worker["surf"], {})
    if binding(current) != worker or current.get("state") in ("exited", "lost"):
        return None
    return {"scope": "retired_planner", "surfaces": [planner], "handoff_to": worker}


def permits_retired(run, body):
    return run is not None and "terminal_state" not in body and retired_body(run) == {
        key: body.get(key) for key in ("scope", "surfaces", "handoff_to")}


def remember(run, surf, incarnation, generation, state=None):
    """Keep the last positively reported binding; absence is never a cleanup instruction."""
    if not isinstance(surf, str) or not surf or len(surf) > 120:
        return
    if type(incarnation) is not int or type(generation) is not int:
        return
    if incarnation < 1 or generation < 1 or generation > run.generation:
        return
    bindings = dict((run.sync_state or {}).get("cleanup_surfaces") or {})
    previous = bindings.get(surf, {})
    if incarnation < previous.get("incarnation", 0) or (
            incarnation == previous.get("incarnation") and generation < previous.get("generation", 0)):
        return
    binding = {"surf": surf, "incarnation": incarnation, "generation": generation}
    if state is not None:
        binding["state"] = state
    elif incarnation == previous.get("incarnation"):
        binding["state"] = previous.get("state", "present")
    if binding != previous:
        bindings[surf] = binding
        run.sync_state = {**(run.sync_state or {}), "cleanup_surfaces": bindings}


def queue(db, run):
    """One stable operation per observed binding/exit phase, including offline delivery."""
    from . import run_coordinator as coordinator

    if not run.host_id:
        return []
    host = db.get(AgentHost, run.host_id)
    if not host or "run.cleanup" not in ((host.caps or {}).get("caps") or []):
        return []
    if run.state not in TERMINAL_RUN_STATES:
        body = retired_body(run)
        if body is None or "run.cleanup.handoff" not in ((host.caps or {}).get("caps") or []):
            return []
        planner = body["surfaces"][0]
        exited = (((run.sync_state or {}).get("cleanup_surfaces") or {}).get(planner["surf"]) or {}).get("state") == "exited"
        identity = json.dumps([run.host_id, run.id, run.generation, body, exited], sort_keys=True)
        op_id = "cleanup-" + str(uuid5(NAMESPACE_URL, identity))
        if db.scalar(select(AgentCommand.id).where(AgentCommand.op_id == op_id)):
            return []
        return [coordinator.command(db, run, "run.cleanup", body, op_id=op_id, source="eridani")]
    bindings = sorted(((run.sync_state or {}).get("cleanup_surfaces") or {}).values(), key=lambda row: row["surf"])
    created = []
    for offset in range(0, len(bindings), 128):
        batch = bindings[offset:offset + 128]
        surfaces = [{key: binding[key] for key in ("surf", "incarnation", "generation")} for binding in batch]
        # A protected terminal-owned process may later actually exit. That positive fact permits a
        # new administrative cleanup, while replaying an unchanged snapshot creates no new work.
        identity = json.dumps([run.id, run.generation, run.state, surfaces,
                               [binding.get("state") == "exited" for binding in batch]], sort_keys=True)
        op_id = "cleanup-" + str(uuid5(NAMESPACE_URL, identity))
        if db.scalar(select(AgentCommand.id).where(AgentCommand.op_id == op_id)):
            continue
        created.append(coordinator.command(db, run, "run.cleanup", {
            "terminal_state": run.state, "surfaces": surfaces}, op_id=op_id, source="eridani"))
    return created


def snapshot(db, owner, host_id, surfaces):
    """A reconnect can discover old ended runs or updated incarnations without guessing."""
    ids = {surface.run_id for surface in surfaces if surface.run_id}
    if not ids:
        return
    runs = {run.id: run for run in db.scalars(select(AgentRun).where(
        AgentRun.owner_id == owner, AgentRun.host_id == host_id, AgentRun.id.in_(ids)).with_for_update())}
    for surface in surfaces:
        run = runs.get(surface.run_id)
        if run:
            remember(run, surface.surf, surface.incarnation, surface.generation, surface.state)
            session = db.get(AgentSession, run.session_pk) if run.session_pk else None
            if surface.state in ("working", "idle", "needs_you") and session and surface.session_id == session.session_id:
                confirm_handoff(run, surface.surf, surface.incarnation, surface.generation)
    for run in runs.values():
        queue(db, run)


def observed(db, run, frame, *, exited=False):
    """A live status/exit can refresh exact facts without waiting for another connection."""
    if getattr(frame, "run_id", None) not in (None, run.id):
        return
    previous = ((run.sync_state or {}).get("cleanup_surfaces") or {}).get(frame.surf, {})
    generation = getattr(frame, "generation", None)
    if generation is None and frame.incarnation == previous.get("incarnation"):
        generation = previous.get("generation")
    remember(run, frame.surf, frame.incarnation, generation, "exited" if exited else None)
    if getattr(frame, "state", None) in ("working", "idle", "needs_you"):
        confirm_handoff(run, frame.surf, frame.incarnation, generation)
    queue(db, run)
