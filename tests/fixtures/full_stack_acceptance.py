"""The real app plus a stand-in for the worker, for the full-stack browser acceptance (scripts/validate_full_stack.py).
Never mounted in production.

Everything else is real: the API, the link to a real remotea server, and that server's agent sessions. What the
worker would do with paid model calls happens here on a thread with scripted models:

- readiness jobs (readiness.py): a task whose title says "migration" changes a schema; one that says "call" is not
  an agent's work; every other task is a small code change with one drafted criterion;
- judge jobs (agent_judge.py): every criterion of the plan is met;
- the supervisor's sweeps for readiness, auto-start and runs. The run sweep is given a clock 70 seconds ahead, so a
  verdict held for a pull request record settles at once instead of after a minute.
"""
import logging
import threading
import time
from datetime import timedelta

from jarvis import agent_judge, auto_start, readiness
from jarvis import run_coordinator as coordinator
from jarvis.api import app  # noqa: F401 - the app uvicorn serves
from jarvis.db import session_scope
from jarvis.models import Job, now
from sqlalchemy import select

log = logging.getLogger("full_stack_acceptance")
readiness.enabled = lambda: True


def ready_answer(body):
    import json

    title = json.loads(body["messages"][1]["content"])["task"]["title"].lower()
    return readiness.Answer(agent_runnable="call" not in title, migration="migration" in title, production_change=False,
                            client_visible=False, spends_money=False, criteria=["The change the task names is made"],
                            reason="Scripted readiness answer.")


def met(evidence):
    return agent_judge.Verdict(verdict="met", gap="", summary="scripted judge", steps=[], criteria=[
        agent_judge.CriterionVerdict(id=c["id"], status="met", evidence="the verify attempt and the changes")
        for c in evidence["plan"]["criteria"]])


def queued(kind):
    with session_scope() as db:
        return list(db.scalars(select(Job.id).where(Job.kind == kind, Job.status == "queued").order_by(Job.created_at)))


def cycle():
    for job in queued(readiness.KIND):
        readiness.process(job, model=ready_answer)
    for job in queued("agent_judge"):
        agent_judge.process(job, model=met)
    for sweep in (readiness.sweep, auto_start.sweep, lambda db: coordinator.sweep(db, now() + timedelta(seconds=70))):
        with session_scope() as db:
            sweep(db)


def work():
    while True:
        try:
            cycle()
        except Exception:  # noqa: BLE001 - a lock met the API (the worker would retry too); keep going
            log.exception("stand-in worker cycle failed")
        time.sleep(0.5)


threading.Thread(target=work, name="stand-in-worker", daemon=True).start()
