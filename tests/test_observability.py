"""Error reporting stays off without a DSN, scrubs what it sends, and the CSP follows it (observability.py)."""

from fastapi.testclient import TestClient
from jarvis import observability
from jarvis.config import get_settings


def test_off_without_a_dsn(monkeypatch):
    import sentry_sdk

    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    monkeypatch.setattr(get_settings(), "sentry_dsn", "")
    assert observability.init_sentry("api") is False
    assert calls == []


def test_on_with_a_dsn_and_private_by_default(monkeypatch):
    import sentry_sdk

    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sentry_sdk, "set_tag", lambda *a: None)
    monkeypatch.setattr(get_settings(), "sentry_dsn", "https://key@o1.ingest.us.sentry.io/2")
    assert observability.init_sentry("worker") is True
    (options,) = calls
    collect = options["data_collection"]
    assert collect["user_info"] is False and collect["stack_frame_variables"] is False
    assert collect["cookies"] == collect["url_query_params"] == {"mode": "off"}
    assert collect["http_headers"] == {"request": {"mode": "off"}} and collect["http_bodies"] == []
    assert options["max_request_body_size"] == "never"
    assert options["traces_sample_rate"] is None
    assert options["before_send"] is observability.scrub_event


def test_scrub_event_drops_credentials_and_bodies():
    event = {
        "request": {
            "url": "https://app.eridani.app/api/v1/auth/google/callback?code=secret&state=s",
            "query_string": "code=secret&state=s",
            "cookies": {"jarvis_session": "token"},
            "data": {"token": "pairing"},
            "env": {"REMOTE_ADDR": "203.0.113.9"},
            "headers": {"Cookie": "jarvis_session=token", "X-CSRF-Token": "csrf", "Authorization": "Bearer k",
                        "CF-Connecting-IP": "203.0.113.9", "User-Agent": "Firefox"},
        },
        "user": {"ip_address": "203.0.113.9", "id": "davin"},
        "breadcrumbs": {"values": [{"category": "httplib", "data": {
            "url": "https://oauth2.googleapis.com/token?x=1", "http.query": "x=1"}}]},
    }
    out = observability.scrub_event(event)
    request = out["request"]
    assert request["url"] == "https://app.eridani.app/api/v1/auth/google/callback"
    assert not {"query_string", "cookies", "data", "env"} & set(request)
    assert request["headers"]["Cookie"] == request["headers"]["X-CSRF-Token"] == "[Filtered]"
    assert request["headers"]["Authorization"] == request["headers"]["CF-Connecting-IP"] == "[Filtered]"
    assert request["headers"]["User-Agent"] == "Firefox"
    assert "ip_address" not in out["user"]
    crumb = out["breadcrumbs"]["values"][0]["data"]
    assert crumb == {"url": "https://oauth2.googleapis.com/token"}


def test_dsn_origin():
    assert observability.dsn_origin("https://abc@o1.ingest.us.sentry.io/42") == "https://o1.ingest.us.sentry.io"
    assert observability.dsn_origin("http://abc@sentry.local:9000/1") == "http://sentry.local:9000"
    for bad in ("", "nonsense", "https://o1.ingest.sentry.io/42", "javascript:alert(1)", "https://abc@host:bad/1"):
        assert observability.dsn_origin(bad) is None


def test_csp_allows_cloudflare_insights_and_sentry_only_when_configured(monkeypatch):
    from jarvis.api import app

    c = TestClient(app)
    monkeypatch.setattr(get_settings(), "sentry_web_dsn", "")
    policy = c.get("/health/live").headers["Content-Security-Policy"]
    directives = {d.split()[0]: d.split()[1:] for d in policy.split("; ")}
    assert "https://static.cloudflareinsights.com" in directives["script-src"]
    assert "https://cloudflareinsights.com" in directives["connect-src"]
    assert "sentry" not in policy
    assert "'unsafe-inline'" not in directives["script-src"]

    monkeypatch.setattr(get_settings(), "sentry_web_dsn", "https://abc@o1.ingest.us.sentry.io/42")
    policy = c.get("/health/live").headers["Content-Security-Policy"]
    assert "https://o1.ingest.us.sentry.io" in policy.split("connect-src ", 1)[1].split(";", 1)[0]


def test_bootstrap_serves_the_browser_dsn(client, monkeypatch):
    assert client.get("/api/v1/bootstrap").json()["sentry_dsn"] is None
    monkeypatch.setattr(get_settings(), "sentry_web_dsn", "https://abc@o1.ingest.us.sentry.io/42")
    assert client.get("/api/v1/bootstrap").json()["sentry_dsn"] == "https://abc@o1.ingest.us.sentry.io/42"
