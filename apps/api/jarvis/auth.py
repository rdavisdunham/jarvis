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
    account_id: str | None = None
    role: str = "owner"


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _authenticate(request: Request, personal=False):
    token = request.cookies.get("jarvis_session", "")
    with session_scope() as db:
        row = db.get(AuthSession, digest(token)) if token else None
        if row is None or row.expires_at <= now() or (row.auth_method == "pairing" and not get_settings().pairing_enabled):
            raise DomainError("NOT_AUTHORIZED", "Sign in to Eridani.", 401)
        from .access import identity as resolve
        from .access import request_access
        namespace, permission = resolve(db,row,personal=personal)
        identity = Identity(namespace,row.device_id,row.csrf,row.owner_id,permission)
        if not personal:
            request_access(request,namespace,permission,row.owner_id)
            device=request.headers.get("X-Device-Id")
            if device and device!=row.device_id:
                raise DomainError("WORKSPACE_CHANGED","Workspace changed in another tab. Reload to continue.",409)
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not secrets.compare_digest(
        request.headers.get("X-CSRF-Token", ""), identity.csrf
    ):
        raise DomainError("NOT_AUTHORIZED", "Refresh this page before trying again.", 403)
    return identity


def authenticate(request: Request):
    return _authenticate(request)


def authenticate_personal(request: Request):
    return _authenticate(request,personal=True)


def sign_in(token):
    settings = get_settings()
    if not settings.pairing_enabled or not settings.owner_token or not secrets.compare_digest(token, settings.owner_token):
        raise DomainError("NOT_AUTHORIZED", "That pairing code is not valid.", 401)
    return new_session(settings.owner_id)


def new_session(owner, method="pairing", *, db=None):
    settings = get_settings()
    session_token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    if db is None:
        with session_scope() as session:
            return new_session(owner, method, db=session)
    from .accounts import ensure_account
    ensure_account(db,owner)
    db.execute(delete(AuthSession).where(AuthSession.expires_at < now()))
    db.add(
        AuthSession(
            token_hash=digest(session_token),
            owner_id=owner,
            device_id=uid(),
            csrf=csrf,
            auth_method=method,
            expires_at=now() + timedelta(hours=settings.session_hours),
        )
    )
    return session_token, csrf
