from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import select

from jarvis import record_reviews as reviews, review_questions as questions, structure
from jarvis.api import app
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, preferences
from jarvis.models import ActionChange, Conversation, ReviewDelivery, WorkspaceMember, now
from jarvis.structure_models import StructureRecord
from test_accounts import client_for, command, shared
from test_external_agents import call, key
from test_organization_foundation import OWNER, apply, create, run, schema


@pytest.fixture(autouse=True)
def crypto(monkeypatch):
    from jarvis.config import get_settings

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())


@pytest.fixture
def clock(monkeypatch):
    """A test clock: move it forward with clock.travel(days=...)."""

    class Clock:
        current = now()

        def travel(self, **delta):
            self.current += timedelta(**delta)
            return self.current

    c = Clock()
    for module in ("jarvis.record_reviews", "jarvis.review_questions", "jarvis.atlas"):
        monkeypatch.setattr(module + ".now", lambda: c.current)
    return c


def cadence(type_id="client", enabled=True, every="1m"):
    d = schema()
    next(t for t in d["types"] if t["id"] == type_id)["review"] = {"enabled": enabled, "every": every}
    return apply(d)


def row(identity):
    with session_scope() as db:
        return db.get(StructureRecord, identity)


def saved(identity):
    with session_scope() as db:
        return structure.data(db, db.get(StructureRecord, identity))


def daytime(clock, hour=10):
    """An instant at `hour` local time on the clock's current day (outside default quiet hours)."""
    with session_scope() as db:
        zone = ZoneInfo(preferences(db, OWNER)["timezone"])
    local = clock.current.astimezone(zone)
    clock.current = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    return clock.current


def queue(instant):
    with session_scope() as db:
        return reviews.queue_due_record_reviews(db, instant)


def inbox():
    with session_scope() as db:
        return [q for q in questions.listing(db, OWNER, status="all")["items"] if q["kind"] == "record_review"]


def test_definition_defaults_off_validates_and_upgrades_legacy_schemas():
    d = schema()
    assert all(t["review"] == {"enabled": False, "every": "1m"} for t in d["types"])
    for bad in ({"enabled": True, "every": "2d"}, {"enabled": "maybe"}, {"enabled": True, "every": "1m", "extra": 1}):
        d = schema()
        next(t for t in d["types"] if t["id"] == "client")["review"] = bad
        with pytest.raises(DomainError):
            apply(d)
    # A legacy stored definition without the setting loads with review off.
    with session_scope() as db:
        s = structure.ensure(db, OWNER)
        legacy = {**s.definition, "types": [{k: v for k, v in t.items() if k != "review"} for t in s.definition["types"]]}
        s.definition = legacy
    assert all(t["review"]["enabled"] is False for t in schema()["types"])
    # Older clients that omit the setting keep the saved cadence.
    cadence("client", every="3m")
    d = schema()
    p = run("structure.preview", dict(expected_revision=d["revision"], definition={
        "types": [{k: v for k, v in t.items() if k != "review"} for t in d["types"]],
        "relationships": d["relationships"]}))
    assert next(t for t in p["definition"]["types"] if t["id"] == "client")["review"] == {"enabled": True, "every": "3m"}


def test_enabling_seeds_spread_next_reviews_and_new_records_follow(clock):
    clients = [create("client", f"Client {i}") for i in range(12)]
    archived = create("client", "Gone")
    run("record.update", dict(record_id=archived["id"], expected_revision=archived["revision"],
                              schema_revision=schema()["revision"], archived=True))
    project = create("project", "Not reviewed")
    start = clock.current
    cadence("client", every="1m")
    dates = [row(c["id"]).next_review_at for c in clients]
    assert all(start + timedelta(days=1) <= d <= start + timedelta(days=32) for d in dates)
    assert len({d.date() for d in dates}) >= 6, "seeding spreads the first reviews over the interval"
    assert row(archived["id"]).next_review_at is None and row(project["id"]).next_review_at is None
    # Seeding is deterministic per record.
    assert dates == [reviews.seeded(c["id"], "1m", start) for c in clients]
    later = create("client", "New client")
    created = row(later["id"])
    assert created.next_review_at == created.created_at + reviews.INTERVALS["1m"]
    assert saved(later["id"])["review_every"] == "1m" and saved(later["id"])["review_due"] is False


def test_due_computation_listing_and_record_payload(clock):
    abc = create("client", "ABC")
    other = create("client", "Other")
    cadence()
    assert reviews_due()["total"] == 0
    with session_scope() as db:
        db.get(StructureRecord, abc["id"]).next_review_at = clock.current - timedelta(hours=1)
        db.get(StructureRecord, other["id"]).next_review_at = clock.current + timedelta(days=3)
    due = reviews_due()
    assert [i["title"] for i in due["items"]] == ["ABC"] and due["total"] == due["due_count"] == 1
    assert due["items"][0]["every_label"] == "every month" and due["items"][0]["due"] is True
    week = reviews_due(within_days=7)
    assert [i["title"] for i in week["items"]] == ["ABC", "Other"] and week["due_count"] == 1
    assert saved(abc["id"])["review_due"] is True and saved(other["id"])["review_due"] is False
    assert "review_queued_at" not in saved(abc["id"])


def reviews_due(**args):
    with session_scope() as db:
        return reviews.listing(db, OWNER, **args)


def overdue(*records, clock):
    with session_scope() as db:
        for r in records:
            db.get(StructureRecord, r["id"]).next_review_at = clock.current - timedelta(days=1)


def test_queue_creates_bounded_idempotent_inbox_items_and_respects_quiet_hours(clock, monkeypatch):
    monkeypatch.setattr(reviews, "BATCH", 3)
    records = [create("client", f"C{i}") for i in range(5)]
    cadence()
    overdue(*records, clock=clock)
    night = daytime(clock, hour=23)
    assert queue(night) == 0 and inbox() == []
    morning = daytime(clock, hour=10)
    assert queue(morning) == 3
    items = inbox()
    assert len(items) == 3 and all(i["status"] == "pending" and i["answer_tool"] == "record.mark_reviewed" for i in items)
    assert items[0]["key"].startswith("review:") and items[0]["record"]["type_name"] == "Client"
    # Once per local day: a second scan the same day adds nothing.
    assert queue(morning + timedelta(hours=3)) == 0 and len(inbox()) == 3
    # The next day the rest are surfaced; already-queued ones are not duplicated.
    clock.travel(days=1)
    assert queue(daytime(clock)) == 2 and len(inbox()) == 5


def test_queue_waits_for_an_active_delivery_lease(clock):
    a = create("client", "ABC")
    cadence()
    overdue(a, clock=clock)
    with session_scope() as db:
        conv = Conversation(owner_id=OWNER, device_id="device", private=False, learning=True)
        db.add(conv)
        db.flush()
        db.add(ReviewDelivery(owner_id=OWNER, device_id="device", conversation_id=conv.id, question_key="field:x",
                              question_revision=1, channel="text", expires_at=daytime(clock) + timedelta(minutes=2)))
    assert queue(daytime(clock)) == 0
    assert queue(daytime(clock) + timedelta(minutes=5)) == 1


def test_mark_reviewed_reschedules_removes_from_inbox_and_reverts(clock):
    a = create("client", "ABC")
    cadence()
    overdue(a, clock=clock)
    queue(daytime(clock))
    [item] = inbox()
    key = str(uuid4())
    with session_scope() as db:
        result = execute(db, OWNER, key, "record.mark_reviewed", item["answer_args"])["data"]
    r = row(a["id"])
    assert r.last_reviewed_at == clock.current and r.next_review_at == clock.current + reviews.INTERVALS["1m"]
    assert result["review_due"] is False and result["revision"] == item["revision"] + 1
    assert inbox() == [] and reviews_due()["total"] == 0
    # Grouped receipts show the review; Revert restores the previous dates and the inbox item.
    from jarvis.action_history import public_change, revert
    with session_scope() as db:
        change = db.scalar(select(ActionChange).where(ActionChange.command_id == key))
        card = public_change(db, change)
        assert card["operation"] == "reviewed" and "Next review" in card["fields"] and card["can_revert"]
        revert(db, OWNER, OWNER, change.id, str(uuid4()))
    assert row(a["id"]).last_reviewed_at is None and len(inbox()) == 1
    # Stale questions are refused.
    with pytest.raises(DomainError):
        run("review.defer", {"question_key": item["key"], "expected_revision": item["revision"], "until": "week"})


def test_undo_toast_restores_review_through_restore_contents(clock):
    a = create("client", "ABC")
    cadence()
    overdue(a, clock=clock)
    queue(daytime(clock))
    [item] = inbox()
    key = str(uuid4())
    with session_scope() as db:
        execute(db, OWNER, key, "record.mark_reviewed", item["answer_args"])
    assert inbox() == []
    # The toast sends the same grouped Undo templates use, keyed by the Mark reviewed command.
    run("record.restore_contents", {"source_command_id": key})
    assert row(a["id"]).last_reviewed_at is None and len(inbox()) == 1
    with session_scope() as db:
        assert db.scalar(select(ActionChange).where(ActionChange.command_id == key)).reverted_by
    with pytest.raises(DomainError):
        run("record.restore_contents", {"source_command_id": key})
    # A later review blocks an older Undo instead of overwriting newer dates.
    second = str(uuid4())
    with session_scope() as db:
        execute(db, OWNER, second, "record.mark_reviewed", {"record_id": a["id"]})
    clock.travel(days=1)
    run("record.mark_reviewed", {"record_id": a["id"]})
    with pytest.raises(DomainError):
        run("record.restore_contents", {"source_command_id": second})


def test_snooze_defers_like_other_questions(clock):
    a, b = create("client", "ABC"), create("client", "Other")
    cadence()
    overdue(a, b, clock=clock)
    queue(daytime(clock))
    first, second = sorted(inbox(), key=lambda q: q["question"])
    run("record.review", {"record_id": first["source_id"], "action": "snooze", "until": "day"})
    run("review.defer", {"question_key": second["key"], "expected_revision": second["revision"], "until": "week"})
    states = {q["question"]: q for q in inbox()}
    assert states["ABC"]["status"] == "deferred" and states["Other"]["status"] == "deferred"
    assert row(a["id"]).next_review_at == clock.current + timedelta(days=1)
    assert row(b["id"]).next_review_at == clock.current + timedelta(days=7)
    with session_scope() as db:
        assert questions.listing(db, OWNER)["counts"]["deferred"] >= 2
    clock.travel(days=2)
    states = {q["question"]: q["status"] for q in inbox()}
    assert states == {"ABC": "pending", "Other": "deferred"}
    with pytest.raises(DomainError):
        run("record.review", {"record_id": a["id"], "action": "snooze"})


def test_pause_and_resume_one_record(clock):
    a = create("client", "ABC")
    cadence()
    overdue(a, clock=clock)
    queue(daytime(clock))
    run("record.review", {"record_id": a["id"], "action": "pause"})
    assert inbox() == [] and reviews_due()["total"] == 0 and row(a["id"]).review_paused
    assert saved(a["id"])["review_due"] is False
    clock.travel(days=1)
    assert queue(daytime(clock)) == 0
    run("record.review", {"record_id": a["id"], "action": "resume"})
    r = row(a["id"])
    assert not r.review_paused and r.next_review_at == clock.current + reviews.INTERVALS["1m"]


def test_disabling_stops_new_items_without_deleting_history(clock):
    a, b = create("client", "ABC"), create("client", "Other")
    cadence()
    overdue(a, b, clock=clock)
    queue(daytime(clock))
    run("record.mark_reviewed", {"record_id": b["id"]})
    reviewed = row(b["id"]).last_reviewed_at
    cadence(enabled=False)
    assert inbox() == [] and reviews_due()["total"] == 0 and saved(a["id"])["review_due"] is False
    clock.travel(days=1)
    assert queue(daytime(clock)) == 0
    assert row(b["id"]).last_reviewed_at == reviewed and row(a["id"]).next_review_at is not None
    with pytest.raises(DomainError) as off:
        run("record.mark_reviewed", {"record_id": a["id"]})
    assert off.value.code == "REVIEW_OFF"
    # Re-enabling reseeds; a recently reviewed record keeps its natural next date.
    cadence(enabled=True)
    assert row(b["id"]).next_review_at == reviewed + reviews.INTERVALS["1m"]
    assert row(a["id"]).next_review_at > clock.current


def test_changing_interval_reschedules():
    a = create("client", "ABC")
    cadence(every="1w")
    assert row(a["id"]).next_review_at <= now() + timedelta(days=7, seconds=5)
    cadence(every="1y")
    assert row(a["id"]).next_review_at > now() + timedelta(days=1)


def test_atlas_includes_review_due_and_type_setting(client, clock):
    a, b = create("client", "ABC"), create("client", "Other")
    cadence()
    overdue(a, clock=clock)
    data = client.get("/api/v1/structure/atlas").json()
    rows = {r["id"]: r for r in data["records"]}
    assert rows[a["id"]]["review_due"] is True and rows[b["id"]]["review_due"] is False
    assert rows[b["id"]]["next_review_at"]
    assert next(t for t in data["types"] if t["id"] == "client")["review"] == {"enabled": True, "every": "1m"}
    assert client.get("/api/v1/structure/reviews").json()["total"] == 1


def test_viewer_reads_but_cannot_mark_reviewed(client):
    guest = client_for("guest", "guest@example.test")
    w = shared(client, guest)
    revision = client.get("/api/v1/structure").json()["revision"]
    made = command(client, "record.create", type_id="client", title="Shared client", schema_revision=revision)
    with session_scope() as db:
        db.get(WorkspaceMember, (w["id"], "guest")).role = "viewer"
    assert guest.get("/api/v1/structure/reviews").status_code == 200
    denied = guest.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "record.mark_reviewed",
                                                  "arguments": {"record_id": made["id"]}})
    assert denied.status_code == 403


def test_bot_scopes(client, clock):
    plain = create("client", "ABC")
    task = create("task", "Weekly check")
    cadence("client")
    cadence("task")
    overdue(plain, task, clock=clock)
    external = TestClient(app)
    _, read_only = key(client, ["records:read"])
    listed = external.get("/api/v1/external/structure/reviews", headers=read_only)
    assert listed.status_code == 200 and {i["title"] for i in listed.json()["items"]} == {"ABC", "Weekly check"}
    assert call(read_only, "record.mark_reviewed", {"record_id": plain["id"]}).status_code == 403
    _, records_only = key(client, ["records:write"])
    assert call(records_only, "record.mark_reviewed", {"record_id": plain["id"]}).status_code == 200
    # Task-backed records follow the core scope rules.
    denied = call(records_only, "record.mark_reviewed", {"record_id": task["id"]})
    assert denied.status_code == 403 and "tasks:write" in denied.text
    _, full = key(client, ["records:write", "tasks:write"])
    assert call(full, "record.mark_reviewed", {"record_id": task["id"]}).status_code == 200


@pytest.mark.asyncio
async def test_eri_tools_list_due_and_mark_reviewed(clock):
    from jarvis.tool_catalog import GROUPS
    from jarvis.tools import READ_TOOLS, _call_tool, registry

    names = {t["name"] for t in registry()}
    assert {"reviews_due", "record_mark_reviewed", "record_review"} <= names
    assert len(GROUPS["review_cadence"][1]) <= 9 and len(GROUPS["records"][1]) <= 9
    assert len(READ_TOOLS["reviews_due"]["description"]) < 300
    a = create("client", "ABC")
    cadence()
    overdue(a, clock=clock)
    listed = await _call_tool(OWNER, "turn", 0, "reviews_due", {"within_days": 7})
    assert [i["title"] for i in listed["items"]] == ["ABC"]
    with session_scope() as db:
        execute(db, OWNER, str(uuid4()), "record.mark_reviewed", {"record_id": a["id"]})
    assert (await _call_tool(OWNER, "turn", 1, "reviews_due", {}))["total"] == 0


def test_record_reviews_stay_out_of_chat_invitations(clock):
    a = create("client", "ABC")
    cadence()
    overdue(a, clock=clock)
    queue(daytime(clock))
    with session_scope() as db:
        conv = Conversation(owner_id=OWNER, device_id="device", private=False, learning=True)
        db.add(conv)
        db.flush()
        assert questions.reserve(db, OWNER, "device", conv.id, "text") is None


def test_preview_names_the_review_change_before_apply():
    create("client", "ABC")
    create("client", "Other")
    d = schema()
    next(t for t in d["types"] if t["id"] == "client")["review"] = {"enabled": True, "every": "3m"}
    p = run("structure.preview", dict(expected_revision=d["revision"],
                                      definition={k: d[k] for k in ("types", "relationships", "field_library", "type_layout")}))
    assert p["impact"]["review_changes"] == [{"type_id": "client", "name": "Client", "every": "3m",
                                              "every_label": "every 3 months", "records": 2}]
    assert all(row_.next_review_at is None for row_ in rows_of("client")), "nothing is scheduled before apply"


def rows_of(type_id):
    with session_scope() as db:
        return list(db.scalars(select(StructureRecord).where(StructureRecord.owner_id == OWNER, StructureRecord.type_id == type_id)))
