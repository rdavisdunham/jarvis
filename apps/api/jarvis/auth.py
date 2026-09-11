import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta

from fastapi import Request
from sqlalchemy import delete

from .config import get_settings
from .db import session_scope
from .domain import DomainError
from .models import AuthSession, now, uid


@dataclass
class Identity:
    owner_id: str
    device_id: str
    csrf: str


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def authenticate(request: Request):
    token = request.cookies.get("jarvis_session", "")
    with session_scope() as db:
        row = db.get(AuthSession, digest(token)) if token else None
        if row is None or row.expires_at <= now():
            raise DomainError("NOT_AUTHORIZED", "Sign in to Eridani.", 401)
        identity = Identity(row.owner_id, row.device_id, row.csrf)
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not secrets.compare_digest(
        request.headers.get("X-CSRF-Token", ""), identity.csrf
    ):
        raise DomainError("NOT_AUTHORIZED", "Refresh this page before trying again.", 403)
    return identity


def sign_in(token):
    settings = get_settings()
    if not settings.owner_token or not secrets.compare_digest(token, settings.owner_token):
        raise DomainError("NOT_AUTHORIZED", "That pairing code is not valid.", 401)
    session_token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with session_scope() as db:
        db.execute(delete(AuthSession).where(AuthSession.expires_at < now()))
        db.add(
            AuthSession(
                token_hash=digest(session_token),
                owner_id=settings.owner_id,
                device_id=uid(),
                csrf=csrf,
                expires_at=now() + timedelta(hours=settings.session_hours),
            )
        )
    return session_token, csrf
