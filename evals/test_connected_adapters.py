import json
from pathlib import Path

import pytest

from scripts.app_eval.connected import PrerequisiteError, execute, google


def test_missing_service_credentials_block_without_network(tmp_path):
    assert execute({"live_config": None, "job": {"target": "google"}}, [])["status"] == "blocked"
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"google": {"dedicated_test_resources": False}}))
    assert execute({"live_config": str(path), "job": {"target": "google"}}, [])["status"] == "blocked"


def test_google_requires_dedicated_calendar_and_cleans_only_created_id(monkeypatch):
    class Settings:
        pass

    monkeypatch.setattr("jarvis.config.get_settings", lambda: Settings())
    for key in ("CLIENT", "SECRET", "REFRESH"):
        monkeypatch.setenv("ERIDANI_EVAL_" + key, "synthetic")
    calls = []

    class Client:
        def __init__(self, *args):
            pass

        def close(self):
            pass

        def request(self, path, **kwargs):
            calls.append((path, kwargs))
            method = kwargs.get("method")
            if method == "DELETE":
                return {}
            if method == "PATCH":
                return {"summary": kwargs["body"]["summary"]}
            if kwargs.get("body"):
                return {"id": kwargs["body"]["id"]}
            if "/events/" in path:
                return {
                    "start": {"date": "2030-01-14"},
                    "end": {"date": "2030-01-15"},
                    "etag": "x",
                    "summary": next(k["body"]["summary"] for _, k in calls if k.get("method") == "PATCH")
                    if any(k.get("method") == "PATCH" for _, k in calls)
                    else "original",
                }
            return {"summary": "Eridani Eval Calendar"}

    monkeypatch.setattr("jarvis.google_calendar.CalendarClient", Client)
    options = {
        "calendar_id": "isolated-calendar",
        "client_id_env": "ERIDANI_EVAL_CLIENT",
        "client_secret_env": "ERIDANI_EVAL_SECRET",
        "refresh_token_env": "ERIDANI_EVAL_REFRESH",
    }
    resources = []
    google(options, resources, lambda: None)
    assert resources[0]["state"] == "removed"
    deleted = [p for p, k in calls if k.get("method") == "DELETE"]
    assert deleted == ["calendars/isolated-calendar/events/" + resources[0]["id"]]
    with pytest.raises(PrerequisiteError, match="secondary"):
        google({**options, "calendar_id": "primary"}, [], lambda: None)
