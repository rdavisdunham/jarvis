from uuid import uuid4

from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.worker import valid_push_endpoint


def test_auth_csrf_and_cross_origin(client):
    payload = {"command_id": str(uuid4()), "tool": "task.create", "arguments": {"title": "Saved"}}
    with TestClient(app) as other:
        assert other.get("/api/v1/tasks").status_code == 401
    assert client.post("/api/v1/commands", json=payload, headers={"X-CSRF-Token": ""}).status_code == 403
    assert (
        client.post(
            "/api/v1/commands", json=payload, headers={"Origin": "https://attacker.example"}
        ).status_code
        == 403
    )
    assert client.post("/api/v1/commands", json=payload).status_code == 200


def test_receipt_export_and_device_conversation_isolation(client):
    command = {"command_id": str(uuid4()), "tool": "task.create", "arguments": {"title": "Hello"}}
    one = client.post("/api/v1/commands", json=command).json()
    two = client.post("/api/v1/commands", json=command).json()
    assert one == two
    assert client.get("/api/v1/commands/" + command["command_id"]).json() == one
    assert client.get("/api/v1/export").json()["tasks"][0]["title"] == "Hello"
    conversation = client.post("/api/v1/conversations", json={}).json()
    with TestClient(app) as other:
        result = other.post("/api/v1/auth/login", json={"token": "test-owner-token"})
        other.headers["X-CSRF-Token"] = result.json()["csrf"]
        assert other.get("/api/v1/conversations/" + conversation["id"]).status_code == 403
    assert client.get("/api/v1/conversations/" + conversation["id"]).json()["messages"] == []


def test_push_destination_restrictions():
    assert valid_push_endpoint("https://fcm.googleapis.com/fcm/send/abc")
    for endpoint in [
        "http://fcm.googleapis.com/send",
        "https://127.0.0.1/admin",
        "https://fcm.googleapis.com.evil.test/send",
        "https://user@fcm.googleapis.com/send",
        "https://fcm.googleapis.com:8443/send",
    ]:
        assert not valid_push_endpoint(endpoint)


def test_private_chat_creation_is_retired_without_silently_saving_it(client):
    rejected = client.post("/api/v1/conversations", json={"private": True})
    assert rejected.status_code == 422
    normal = client.post("/api/v1/conversations", json={}).json()
    assert not normal["private"]
    assert client.post("/api/v1/conversations", json={"private": False}).status_code == 200
    saved = client.post("/api/v1/commands", json={
        "command_id": str(uuid4()), "tool": "settings.update",
        "arguments": {"history_enabled": False},
    })
    assert saved.status_code == 200
    assert client.post("/api/v1/conversations", json={}).json()["private"]
