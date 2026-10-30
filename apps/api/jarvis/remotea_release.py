"""Pairs a linked remotea's version with its latest GitHub release, for the "update available" note in Settings.

The capability negotiation tolerates a host one release behind, so the note appears only when the skew is larger:
more than one minor version, or any major difference. The release is looked up at most every six hours, off the
request path (a background thread fills the cache); a failed lookup, or no published release (404), shows no note.
"""

import logging
import os
import re
import threading
import time

import httpx

from .config import get_settings

LATEST_URL = "https://api.github.com/repos/rdavisdunham/remotea/releases/latest"
REFRESH_SECONDS = 6 * 3600
FAILURE_SECONDS = 15 * 60
VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?")

log = logging.getLogger(__name__)
_lock = threading.Lock()
# {"checked": monotonic seconds, "version": "0.3.0" | None, "url": str | None}; empty until the first lookup ends.
_cache: dict = {}
_inflight = False


def parse(version):
    match = VERSION_RE.match(version or "")
    return tuple(int(part or 0) for part in match.groups()) if match else None


def behind(host_version, latest_version):
    """True when the host trails the release by more than one minor version (or a major one)."""
    have, want = parse(host_version), parse(latest_version)
    if have is None or want is None or have >= want:
        return False
    return want[0] > have[0] or want[1] - have[1] > 1


def latest():
    """The cached release ({version, url}) or None; starts a background refresh when stale."""
    with _lock:
        stale = not _cache or time.monotonic() - _cache["checked"] > (
            REFRESH_SECONDS if _cache.get("ok") else FAILURE_SECONDS)
        cached = {k: _cache.get(k) for k in ("version", "url")} if _cache.get("version") else None
    if stale and enabled():
        refresh_in_background()
    return cached


def update_for(host_version):
    """{latest, url} when the host should update, else None."""
    release = latest()
    if release and behind(host_version, release["version"]):
        return {"latest": release["version"], "url": release["url"]}
    return None


def enabled():
    return get_settings().external_services_enabled and "PYTEST_CURRENT_TEST" not in os.environ


def refresh_in_background():
    global _inflight
    with _lock:
        if _inflight:
            return
        _inflight = True
    threading.Thread(target=refresh, name="remotea-release", daemon=True).start()


def refresh(client=None):
    """One lookup; the result (or the failure) is cached either way."""
    global _inflight
    result = {"version": None, "url": None, "ok": False}
    try:
        response = (client or httpx).get(LATEST_URL, timeout=10, headers={"Accept": "application/vnd.github+json"})
        if response.status_code == 404:
            result["ok"] = True  # No release published yet: nothing to pair with.
        else:
            response.raise_for_status()
            body = response.json()
            tag = str(body.get("tag_name") or "")
            if parse(tag):
                result.update(version=tag.lstrip("v"), url=str(body.get("html_url") or "")[:500] or None, ok=True)
    except (httpx.HTTPError, ValueError) as exc:
        log.info("remotea release lookup failed: %s", exc)
    with _lock:
        _cache.clear()
        _cache.update(result, checked=time.monotonic())
        _inflight = False
    return result


def set_cache(version, url=None):
    """Tests: as if a lookup had just returned this release (None: no release published)."""
    with _lock:
        _cache.clear()
        _cache.update(version=version, url=url, ok=True, checked=time.monotonic())
