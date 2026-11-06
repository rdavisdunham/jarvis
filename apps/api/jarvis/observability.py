"""Error reporting (Sentry) and the page CSP that depends on it.

Everything here is off unless JARVIS_SENTRY_DSN (API and worker) or JARVIS_SENTRY_WEB_DSN (browser) is set.
Reports carry the exception and stack, never cookies, auth or CSRF headers, query strings, request bodies or
stack-frame locals: those can hold sign-in material and the owner's planner data.
"""

import logging
import os
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

SENSITIVE_HEADERS = frozenset({
    "authorization", "proxy-authorization", "cookie", "set-cookie", "x-csrf-token", "x-device-id",
    "x-api-key", "cf-connecting-ip", "x-forwarded-for", "x-real-ip", "true-client-ip",
})
REDACTED = "[Filtered]"
# sentry-sdk's data_collection option (it supersedes send_default_pii and include_local_variables).
DATA_COLLECTION = {
    "user_info": False,
    "cookies": {"mode": "off"},
    "http_headers": {"request": {"mode": "off"}},
    "http_bodies": [],
    "url_query_params": {"mode": "off"},
    "stack_frame_variables": False,
    "gen_ai": {"inputs": False, "outputs": False},
    "database_query_data": False,
    "queues": False,
}


def scrub_event(event, hint=None):
    """Sentry before_send: drop credentials, bodies and addresses before anything leaves the process."""
    request = event.get("request")
    if isinstance(request, dict):
        for key in ("cookies", "data", "query_string", "env"):
            request.pop(key, None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            request["headers"] = {
                name: (REDACTED if name.lower() in SENSITIVE_HEADERS else value) for name, value in headers.items()
            }
        url = request.get("url")
        if isinstance(url, str) and ("?" in url or "#" in url):
            request["url"] = url.split("?", 1)[0].split("#", 1)[0]
    user = event.get("user")
    if isinstance(user, dict):
        user.pop("ip_address", None)
    crumbs = event.get("breadcrumbs")
    for crumb in (crumbs.get("values") if isinstance(crumbs, dict) else crumbs) or []:
        scrub_breadcrumb(crumb)
    return event


def scrub_breadcrumb(crumb, hint=None):
    data = crumb.get("data") if isinstance(crumb, dict) else None
    if isinstance(data, dict):
        for key in ("http.query", "http.fragment"):
            data.pop(key, None)
        url = data.get("url")
        if isinstance(url, str):
            data["url"] = url.split("?", 1)[0].split("#", 1)[0]
    return crumb


def init_sentry(component: str) -> bool:
    """Start Sentry for this process when JARVIS_SENTRY_DSN is set; returns whether it started."""
    from .config import get_settings

    settings = get_settings()
    dsn = settings.sentry_dsn.strip()
    if not dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=settings.deployment_environment,
            release=(os.getenv("RAILWAY_GIT_COMMIT_SHA") or "")[:12] or None,
            # Collect nothing about the person or the request beyond method and path; scrub_event is the backstop.
            data_collection=DATA_COLLECTION,
            max_request_body_size="never",
            # Errors only: no performance tracing or profiling (no added cost).
            traces_sample_rate=None,
            before_send=scrub_event,
            before_breadcrumb=scrub_breadcrumb,
        )
        sentry_sdk.set_tag("component", component)
        return True
    except Exception:  # noqa: BLE001 - reporting must never stop the app; the DSN is not logged
        logger.warning("Error reporting could not start; continuing without it.")
        return False


def dsn_origin(dsn: str) -> str | None:
    """The ingest origin a browser DSN posts to (for connect-src), or None when unset or malformed."""
    try:
        parts = urlsplit((dsn or "").strip())
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in {"https", "http"} or not parts.hostname or not parts.username:
        return None
    host = parts.hostname if ":" not in parts.hostname else f"[{parts.hostname}]"
    return f"{parts.scheme}://{host}" + (f":{port}" if port else "")


# Cloudflare Web Analytics: the proxied zone injects beacon.min.js from static.cloudflareinsights.com,
# which reports to cloudflareinsights.com.
INSIGHTS_SCRIPT = "https://static.cloudflareinsights.com"
INSIGHTS_CONNECT = "https://cloudflareinsights.com"


def content_security_policy(settings) -> str:
    connect = ["'self'", "https://api.openai.com", "wss://api.openai.com", INSIGHTS_CONNECT]
    ingest = dsn_origin(settings.sentry_web_dsn)
    if ingest:
        connect.append(ingest)
    return (
        f"default-src 'self'; script-src 'self' {INSIGHTS_SCRIPT}; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; media-src 'self' blob:; "
        f"connect-src {' '.join(connect)}; frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
    )
