"""Stage E: collector cutover readiness — seven consecutive clean parity days (agent_link.readiness)."""

from datetime import timedelta

from fastapi.testclient import TestClient
from jarvis import agent_link
from jarvis.api import app
from jarvis.db import session_scope
from jarvis.models import now
from jarvis.session_models import AgentParityDay, AgentSession
from sqlalchemy import select
from test_agent_link import SESSION, bearer, enable, report

OWNER = "davin"


def days(*outcomes, end=0):
    """Parity days ending `end` days ago; each outcome is (reports, mismatched, missing) or None for no reports."""
    with session_scope() as db:
        today = agent_link.owner_today(db, OWNER)
        for back, outcome in enumerate(reversed(outcomes)):
            if outcome is None:
                continue
            reports, mismatched, missing = outcome
            db.add(AgentParityDay(owner_id=OWNER, day=today - timedelta(days=back + end), reports=reports,
                                  matched=reports - mismatched - missing, mismatched=mismatched, missing=missing,
                                  updated_at=now()))


def ready():
    with session_scope() as db:
        return agent_link.readiness(db, OWNER)


CLEAN = (5, 0, 0)


def test_seven_clean_days_ending_today_or_yesterday_are_ready(client):
    days(*[CLEAN] * 6)
    assert ready()["streak"] == 6 and not ready()["ready"]
    with session_scope() as db:
        db.execute(AgentParityDay.__table__.delete())
    days(*[CLEAN] * 7, end=1)  # Today has no reports yet: yesterday ends the streak.
    result = ready()
    assert result["ready"] and result["streak"] == 7 and result["needed"] == 7
    assert result["days"][0]["reports"] == 0 and all(d["clean"] for d in result["days"][1:])


def test_a_mismatch_or_a_silent_day_breaks_the_streak_and_missing_does_not(client):
    days(*[CLEAN] * 3, (4, 1, 0), *[CLEAN] * 3)
    assert ready()["streak"] == 3 and not ready()["ready"]
    with session_scope() as db:
        db.execute(AgentParityDay.__table__.delete())
    days(*[CLEAN] * 3, None, *[CLEAN] * 3)
    assert ready()["streak"] == 3
    with session_scope() as db:
        db.execute(AgentParityDay.__table__.delete())
    days(*[(5, 0, 2)] * 7)  # Sessions only the new collector saw are not a regression.
    assert ready()["ready"]


def test_parity_reports_count_toward_the_owners_day_and_the_summary_shows_readiness(client):
    token = enable(client)["token"]
    headers = bearer(token)
    report(headers, session_id="s1", **SESSION)
    with session_scope() as db:
        stored = db.scalar(select(AgentSession))
        body = {**SESSION, "last_active_at": stored.last_active_at.isoformat()}
    external = TestClient(app)
    for change in ({}, {}, {"branch": "other"}):
        external.post("/api/v1/external/agents/parity", headers=headers,
                      json={"tool": "claude_code", "session_id": "s1", "report": {**body, **change}})
    external.post("/api/v1/external/agents/parity", headers=headers,
                  json={"tool": "claude_code", "session_id": "s-new", "report": body})
    with session_scope() as db:
        [row] = db.scalars(select(AgentParityDay))
        assert (row.reports, row.matched, row.mismatched, row.missing) == (4, 2, 1, 1)
        assert row.day == agent_link.owner_today(db, OWNER)
    summary = client.get("/api/v1/remotea/parity").json()
    assert summary["cutover"]["ready"] is False and summary["cutover"]["streak"] == 0
    assert summary["cutover"]["days"][0] == {"day": str(row.day), "reports": 4, "matched": 2, "mismatched": 1,
                                             "missing": 1, "clean": False}
