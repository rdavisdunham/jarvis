"""Rate-limit keys behind Railway's proxy and Cloudflare (client_ip.py)."""

import pytest
from fastapi.testclient import TestClient
from jarvis.client_ip import client_ip
from starlette.requests import Request

CLOUDFLARE_EDGE = "172.70.1.2"  # inside 172.64.0.0/13
RAILWAY_PEER = "100.64.0.3"


def request(peer=RAILWAY_PEER, headers=()):
    return Request({
        "type": "http", "method": "POST", "path": "/", "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": (peer, 1234) if peer else None,
    })


@pytest.fixture
def trust_all(monkeypatch):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")


def test_through_cloudflare_uses_cf_connecting_ip(trust_all):
    # The browser wrote 6.6.6.6; Cloudflare appended the real address; Railway appended Cloudflare's edge.
    r = request(headers=[("X-Forwarded-For", f"6.6.6.6, 203.0.113.9, {CLOUDFLARE_EDGE}"),
                         ("CF-Connecting-IP", "203.0.113.9")])
    assert client_ip(r) == "203.0.113.9"


def test_spoofed_forwarded_for_is_ignored(trust_all):
    # uvicorn with "*" would have used the leftmost (browser-written) entry.
    for spoof in ("1.1.1.1", "8.8.8.8", "203.0.113.200"):
        r = request(peer=spoof, headers=[("X-Forwarded-For", f"{spoof}, 198.51.100.7")])
        assert client_ip(r) == "198.51.100.7"


def test_cf_header_from_outside_cloudflare_is_ignored(trust_all):
    # Sent straight to Railway, bypassing Cloudflare: the header is attacker-controlled.
    r = request(headers=[("X-Forwarded-For", "198.51.100.7"), ("CF-Connecting-IP", "203.0.113.9")])
    assert client_ip(r) == "198.51.100.7"


def test_ipv6_and_ports(trust_all):
    r = request(headers=[("X-Forwarded-For", "[2606:4700::6810:1]:443"), ("CF-Connecting-IP", "2001:db8::5")])
    assert client_ip(r) == "2001:db8::5"
    r = request(headers=[("X-Forwarded-For", "198.51.100.7:5555")])
    assert client_ip(r) == "198.51.100.7"
    r = request(headers=[("X-Forwarded-For", "::ffff:198.51.100.7")])
    assert client_ip(r) == "198.51.100.7"


def test_garbage_cf_header_falls_back_to_the_edge(trust_all):
    r = request(headers=[("X-Forwarded-For", CLOUDFLARE_EDGE), ("CF-Connecting-IP", "not-an-ip")])
    assert client_ip(r) == CLOUDFLARE_EDGE


def test_default_trust_keeps_the_peer_uvicorn_resolved(monkeypatch):
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    # Not trusted: uvicorn leaves the raw peer, and neither header is believed.
    r = request(peer="198.51.100.7", headers=[("X-Forwarded-For", "1.1.1.1"), ("CF-Connecting-IP", "1.1.1.1")])
    assert client_ip(r) == "198.51.100.7"
    # A specific allow-list: uvicorn resolved the first untrusted hop, a Cloudflare edge.
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "100.64.0.0/10")
    r = request(peer=CLOUDFLARE_EDGE, headers=[("CF-Connecting-IP", "203.0.113.9")])
    assert client_ip(r) == "203.0.113.9"


def test_no_address():
    assert client_ip(request(peer=None)) == "unknown"


def test_login_limit_cannot_be_dodged_by_rotating_forwarded_for(trust_all):
    from jarvis.api import app, login_attempts

    login_attempts.clear()
    c = TestClient(app)
    for n in range(10):
        r = c.post("/api/v1/auth/login", json={"token": "wrong"},
                   headers={"X-Forwarded-For": f"10.0.0.{n}, {CLOUDFLARE_EDGE}", "CF-Connecting-IP": "203.0.113.9"})
        assert r.status_code == 401
    r = c.post("/api/v1/auth/login", json={"token": "wrong"},
               headers={"X-Forwarded-For": f"10.9.9.9, {CLOUDFLARE_EDGE}", "CF-Connecting-IP": "203.0.113.9"})
    assert r.status_code == 429
    # Someone else behind the same Cloudflare edge is not locked out.
    r = c.post("/api/v1/auth/login", json={"token": "test-owner-token"},
               headers={"X-Forwarded-For": f"203.0.113.50, {CLOUDFLARE_EDGE}", "CF-Connecting-IP": "203.0.113.50"})
    assert r.status_code == 200
    login_attempts.clear()
