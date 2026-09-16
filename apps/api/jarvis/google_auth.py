"""Verified Google identities: existing links or explicit email-bound invitations."""

import json
import secrets
from datetime import timedelta
from urllib.parse import urlsplit

import httpx
from cryptography.fernet import Fernet, InvalidToken
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from sqlalchemy import delete, select

from .auth import digest, new_session
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit
from .models import AuthSession, GoogleCalendar, GoogleCalendarEvent, GoogleIdentity, GoogleOAuthAttempt, now

IDENTITY_SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email"]
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
WRITE_SCOPE = "https://www.googleapis.com/auth/calendar.events"
COOKIE = "eri_google_oauth"
CALLBACK_PATH = "/api/v1/auth/google/callback"


def configured():
    settings = get_settings()
    try:
        Fernet(settings.integration_encryption_key.encode())
    except (ValueError, TypeError):
        return False
    return bool(settings.google_client_id and settings.google_client_secret)


def callback_uri():
    return get_settings().origin.rstrip("/") + CALLBACK_PATH


def cipher():
    try:
        return Fernet(get_settings().integration_encryption_key.encode())
    except (ValueError, TypeError):
        raise DomainError("INTEGRATION_UNAVAILABLE", "Google connection needs server setup.", 503) from None


def seal(value):
    return cipher().encrypt(json.dumps(value).encode()).decode()


def unseal(value):
    try:
        return json.loads(cipher().decrypt(value.encode()))
    except (InvalidToken, ValueError, TypeError):
        raise DomainError("GOOGLE_RECONNECT", "Reconnect Google in Settings.", 409) from None


def make_flow(purpose, state, verifier):
    settings = get_settings()
    return Flow.from_client_config(
        {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=IDENTITY_SCOPES
        + ([CALENDAR_SCOPE] if purpose in {"calendar", "calendar_write"} else [])
        + ([WRITE_SCOPE] if purpose == "calendar_write" else []),
        state=state,
        code_verifier=verifier,
        autogenerate_code_verifier=False,
        redirect_uri=callback_uri(),
    )


def begin(purpose, session_hash=None):
    if purpose != "login":
        from .config import require_external_services
        require_external_services()
    if not configured():
        raise DomainError("INTEGRATION_UNAVAILABLE", "Google connection needs server setup.", 503)
    parsed = urlsplit(callback_uri())
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise DomainError("INTEGRATION_UNAVAILABLE", "Google sign-in requires a secure app address.", 503)
    if purpose not in {"login", "link", "calendar", "calendar_write"}:
        raise DomainError("INVALID_ARGUMENT", "Choose a supported Google connection.")
    state, browser, nonce, verifier = [secrets.token_urlsafe(48) for _ in range(4)]
    with session_scope() as db:
        if purpose != "login":
            session = db.get(AuthSession, session_hash)
            if not session or session.expires_at <= now():
                raise DomainError("NOT_AUTHORIZED", "Pair this device before linking Google.", 401)
        db.execute(delete(GoogleOAuthAttempt).where(GoogleOAuthAttempt.expires_at < now()))
        account = db.get(GoogleIdentity, session.owner_id) if purpose != "login" else None
        db.add(
            GoogleOAuthAttempt(
                state_hash=digest(state),
                browser_hash=digest(browser),
                purpose=purpose,
                session_hash=session_hash,
                account_subject=account.subject if account else None,
                account_generation=account.generation if account else None,
                nonce=nonce,
                verifier=seal(verifier),
                expires_at=now() + timedelta(minutes=10),
            )
        )
    flow = make_flow(purpose, state, verifier)
    kwargs = {"nonce": nonce, "prompt": "select_account"}
    if purpose in {"calendar", "calendar_write"}:
        kwargs.update(access_type="offline", prompt="consent select_account", include_granted_scopes="true")
    url, _ = flow.authorization_url(**kwargs)
    return url, browser


def exchange(purpose, state, verifier, code):
    if purpose != "login":
        from .config import require_external_services
        require_external_services()
    flow = make_flow(purpose, state, verifier)
    # This session is local to one exchange; partial/incremental grants are checked explicitly below.
    flow.oauth2session._client.scope = None
    flow.oauth2session.scope = None
    return flow.fetch_token(code=code, timeout=20)


class VerificationRequest(GoogleRequest):
    def __call__(self, *args, **kwargs):
        kwargs["timeout"] = 15
        return super().__call__(*args, **kwargs)


def verify_identity(token):
    return id_token.verify_oauth2_token(
        token, VerificationRequest(), get_settings().google_client_id, clock_skew_in_seconds=30
    )


def finish(state, browser, code=None, error=None):
    if not state or not browser:
        raise DomainError("GOOGLE_STATE", "That Google sign-in expired. Start again.", 400)
    with session_scope() as db:
        attempt = db.get(GoogleOAuthAttempt, digest(state), with_for_update=True)
        if (
            not attempt
            or attempt.expires_at <= now()
            or not secrets.compare_digest(attempt.browser_hash, digest(browser))
        ):
            raise DomainError("GOOGLE_STATE", "That Google sign-in expired. Start again.", 400)
        values = {
            key: getattr(attempt, key)
            for key in [
                "purpose",
                "session_hash",
                "nonce",
                "verifier",
                "account_subject",
                "account_generation",
            ]
        }
        db.delete(attempt)  # One use, including denied/failed exchanges.
    if error:
        raise DomainError("GOOGLE_DENIED", "Google connection was cancelled. Nothing changed.")
    if not code:
        raise DomainError("GOOGLE_STATE", "Google did not return a sign-in code.")
    try:
        tokens = exchange(values["purpose"], state, unseal(values["verifier"]), code)
        claims = verify_identity(tokens["id_token"])
    except Exception as exc:
        # Provider errors can contain tokens or authorization codes. Never reflect or log them.
        if isinstance(exc, DomainError):
            raise
        raise DomainError(
            "GOOGLE_SIGN_IN", "Google sign-in could not finish. Please try again.", 502
        ) from None
    subject, email = claims.get("sub"), claims.get("email")
    if (
        not isinstance(subject, str)
        or not subject
        or not isinstance(email, str)
        or not email
        or claims.get("email_verified") is not True
        or not secrets.compare_digest(str(claims.get("nonce", "")), values["nonce"])
    ):
        raise DomainError("GOOGLE_IDENTITY", "Google could not verify this account.", 401)
    with session_scope() as db:
        from .accounts import ensure_account
        from .models import WorkspaceInvite, uid
        purpose=values["purpose"]
        # Serialize a verified subject through first-account creation and linking.
        advisory(db,f"google-subject:{subject}")
        linked=db.scalar(select(GoogleIdentity).where(GoogleIdentity.subject==subject))
        if purpose=="login":
            if linked:
                owner=linked.owner_id
            else:
                invited=db.scalar(select(WorkspaceInvite).where(
                    WorkspaceInvite.email==email.casefold(),WorkspaceInvite.status=="pending",WorkspaceInvite.expires_at>now()))
                if not invited:
                    raise DomainError("GOOGLE_ACCOUNT","This Google account needs an invitation to Eridani.",403)
                owner=uid()
                ensure_account(db,owner,email.split("@")[0])
        else:
            session=db.get(AuthSession,values["session_hash"])
            if not session or session.expires_at<=now():
                raise DomainError("NOT_AUTHORIZED","Sign in and start the connection again.",401)
            owner=session.owner_id
            if session.workspace_id:
                raise DomainError("PERSONAL_WORKSPACE","Connect Google from your personal workspace.",403)
            if linked and linked.owner_id!=owner:
                raise DomainError("GOOGLE_ACCOUNT","That Google account is already linked to another user.",403)
        advisory(db, f"google:{owner}")
        identity = db.get(GoogleIdentity, owner)
        if purpose != "login":
            session = db.get(AuthSession, values["session_hash"])
            if not session or session.owner_id != owner or session.expires_at <= now():
                raise DomainError("NOT_AUTHORIZED", "Your pairing session ended. Sign in and try again.", 401)
        if values["account_subject"] is not None and (
            not identity
            or identity.subject != values["account_subject"]
            or (purpose != "login" and identity.generation != values["account_generation"])
        ):
            raise DomainError("GOOGLE_STATE", "Google connection changed. Start again.", 409)
        if identity and identity.subject != subject:
            raise DomainError("GOOGLE_ACCOUNT", "Use the Google account already linked to Eridani.", 403)
        if not identity:
            identity = GoogleIdentity(owner_id=owner, subject=subject, email=email)
            db.add(identity)
            db.flush()
        identity.email = email
        if purpose in {"calendar", "calendar_write"}:
            scopes = tokens.get("scope", [])
            scopes = scopes.split() if isinstance(scopes, str) else scopes
            if CALENDAR_SCOPE not in scopes:
                raise DomainError(
                    "GOOGLE_SCOPE", "Calendar permission was not granted. Sign-in is unchanged.", 403
                )
            if purpose == "calendar_write" and WRITE_SCOPE not in scopes:
                raise DomainError(
                    "GOOGLE_SCOPE",
                    "Calendar editing permission was not granted. Existing access is unchanged.",
                    403,
                )
            refresh = tokens.get("refresh_token")
            if not refresh and identity.credentials:
                try:
                    refresh = unseal(identity.credentials).get("refresh_token")
                except DomainError:
                    pass
            if not refresh:
                raise DomainError(
                    "GOOGLE_RECONNECT", "Reconnect Calendar and allow access while you are away.", 409
                )
            identity.credentials = seal({"refresh_token": refresh})
            identity.calendar_write_enabled = WRITE_SCOPE in scopes
            identity.calendar_enabled, identity.status, identity.error = True, "pending", ""
            identity.generation += 1
            identity.next_sync_at = now()
        emit(db, owner, "google.changed", owner)
        # Keep issuance under the same lock/transaction so unlink cannot precede a late session.
        return new_session(owner, "google", db=db)


def disconnect_calendar(owner, *, unlink=False):
    with session_scope() as db:
        advisory(db, f"google:{owner}")
        identity = db.get(GoogleIdentity, owner)
        if not identity:
            return {"disconnected": True, "revoked": True}
        token = None
        if identity.credentials:
            try:
                token = unseal(identity.credentials).get("refresh_token")
            except DomainError:
                pass
        identity.credentials, identity.calendar_enabled, identity.calendar_write_enabled = None, False, False
        identity.status, identity.error, identity.last_sync_at, identity.next_sync_at = (
            "not_connected",
            "",
            None,
            None,
        )
        identity.generation += 1
        ids = select(GoogleCalendar.id).where(GoogleCalendar.owner_id == owner)
        db.execute(delete(GoogleCalendarEvent).where(GoogleCalendarEvent.calendar_id.in_(ids)))
        db.execute(delete(GoogleCalendar).where(GoogleCalendar.owner_id == owner))
        if unlink:
            db.execute(
                delete(AuthSession).where(AuthSession.owner_id == owner, AuthSession.auth_method == "google")
            )
            hashes=select(AuthSession.token_hash).where(AuthSession.owner_id==owner)
            db.execute(delete(GoogleOAuthAttempt).where(GoogleOAuthAttempt.session_hash.in_(hashes)))
            db.delete(identity)
        emit(db, owner, "google.changed", owner)
    revoked = True
    if token and get_settings().external_services_enabled:
        try:
            result = httpx.post("https://oauth2.googleapis.com/revoke", data={"token": token}, timeout=15)
            revoked = result.status_code in {200, 400}
        except httpx.HTTPError:
            revoked = False
    return {"disconnected": True, "revoked": revoked}


def unlink_google(owner):
    if owner!=get_settings().owner_id:
        raise DomainError("LOGIN_REQUIRED","Google is your sign-in method. Disconnect Calendar instead to keep account access.",409)
    return {**disconnect_calendar(owner, unlink=True), "unlinked": True}
