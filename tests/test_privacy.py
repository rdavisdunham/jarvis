from uuid import uuid4

from jarvis.db import session_scope
from jarvis.domain import capture_source, execute, owned
from jarvis.models import Command, Conversation, Memory, Source
from sqlalchemy import select


def cmd(db, tool, arguments):
    return execute(db, "davin", str(uuid4()), tool, arguments)


def test_deleted_source_redacts_memory_command_receipts(client):
    with session_scope() as db:
        result = cmd(db, "memory.capture", {"content": "Sensitive preference for a test"})
        memory_id = result["data"]["id"]
        command_id = result["command_id"]
        cmd(db, "memory.forget", {"memory_id": memory_id, "delete_source": True})
    with session_scope() as db:
        assert db.get(Memory, memory_id).content == ""
        receipt = db.get(Command, ("davin", command_id))
        assert receipt.result["data"] == {"id": memory_id, "deleted": True}
        assert all(s.content == "" for s in db.scalars(select(Source)))
    assert "Sensitive preference" not in client.get("/api/v1/export").text


def test_history_setting_applies_to_existing_conversation(client):
    data = client.post("/api/v1/conversations", json={"private": False}).json()
    with session_scope() as db:
        conv = owned(db, Conversation, data["id"], "davin")
        assert capture_source(db, "davin", "Before", "before", conversation=conv) is not None
        cmd(db, "settings.update", {"history_enabled": False})
        assert capture_source(db, "davin", "After", "after", conversation=conv) is None
        assert (
            capture_source(db, "davin", "Save explicitly", "explicit", conversation=conv, explicit=True)
            is not None
        )


def test_untrusted_csv_cells_and_malformed_push_endpoint(client):
    client.post(
        "/api/v1/commands",
        json={"command_id": str(uuid4()), "tool": "task.create", "arguments": {"title": "=1+1"}},
    )
    assert "'=1+1" in client.get("/api/v1/export?format=csv").text
    result = client.post(
        "/api/v1/push",
        json={"endpoint": "https://fcm.googleapis.com:bad/send", "keys": {"auth": "a", "p256dh": "b"}},
    )
    assert result.status_code == 400
