"""Outbound calendar synchronization. Provider tokens never leave the server."""

from datetime import datetime, timedelta
from urllib.parse import quote

import httpx
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials
from sqlalchemy import delete, select

from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, check_revision, emit, enqueue_job, owned
from .google_auth import VerificationRequest, configured, unseal
from .models import GoogleCalendar, GoogleCalendarEvent, GoogleIdentity, Job, now

POLL_PENDING = {"queued", "running", "retrying"}


class SyncFailure(Exception):
    def __init__(self, code, status=0):
        self.code, self.status = code, status
        super().__init__(code)


class CalendarClient:
    def __init__(self, credentials):
        settings = get_settings()
        token = Credentials(
            None,
            refresh_token=credentials["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
        )
        try:
            token.refresh(VerificationRequest())
        except RefreshError:
            raise SyncFailure("reconnect", 401) from None
        except TransportError:
            raise SyncFailure("unavailable") from None
        self.client = httpx.Client(
            base_url="https://www.googleapis.com/calendar/v3/",
            headers={"Authorization": "Bearer " + token.token},
            timeout=20,
            follow_redirects=False,
        )

    def request(self, path, params=None, body=None, *, method=None, headers=None):
        try:
            response = self.client.request(
                (method or ("POST" if body is not None else "GET")),
                path,
                params=params,
                json=body,
                headers=headers,
            )
            if response.status_code >= 400:
                code = (
                    "reconnect"
                    if response.status_code == 401
                    else "rate_limited"
                    if response.status_code == 429
                    else "unavailable"
                )
                raise SyncFailure(code, response.status_code)
            return {} if response.status_code == 204 else response.json()
        except (httpx.HTTPError, ValueError):
            raise SyncFailure("unavailable") from None

    def close(self):
        self.client.close()


def connection_status(db, owner):
    account = db.get(GoogleIdentity, owner)
    rows = list(
        db.scalars(
            select(GoogleCalendar).where(GoogleCalendar.owner_id == owner).order_by(GoogleCalendar.title)
        )
    )
    pending = (
        bool(
            db.scalar(
                select(Job.id)
                .where(Job.owner_id == owner, Job.kind == "google_sync", Job.status.in_(POLL_PENDING))
                .limit(1)
            )
        )
        if account and account.calendar_enabled
        else False
    )
    return {
        "configured": configured(),
        "linked": bool(account),
        "email": account.email if account else None,
        "calendar_enabled": bool(account and account.calendar_enabled),
        "calendar_write_enabled": bool(account and account.calendar_write_enabled),
        "status": account.status if account else "not_connected",
        "syncing": pending,
        "error": account.error if account else "",
        "last_sync_at": account.last_sync_at.isoformat() if account and account.last_sync_at else None,
        "stale": not account
        or not account.last_sync_at
        or now() - account.last_sync_at > timedelta(minutes=15),
        "calendars": [
            {
                "id": row.id,
                "title": row.title,
                "timezone": row.timezone,
                "selected": row.selected,
                "available": row.available,
                "primary": row.primary,
                "access_role": row.access_role,
                "writable": bool(
                    account
                    and account.calendar_write_enabled
                    and row.access_role in {"owner", "writer"}
                    and row.available
                    and row.selected
                ),
                "revision": row.revision,
                "last_sync_at": row.last_sync_at.isoformat() if row.last_sync_at else None,
            }
            for row in rows
        ],
    }


def queue_sync(db, owner, force=False):
    advisory(db, f"google:{owner}")
    account = db.get(GoogleIdentity, owner)
    if not configured() or not account or not account.calendar_enabled or account.status == "needs_reconnect":
        return None
    pending = db.scalar(
        select(Job)
        .where(Job.owner_id == owner, Job.kind == "google_sync", Job.status.in_(POLL_PENDING))
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    if pending:
        if now() - pending.created_at < timedelta(minutes=15):
            return pending.id
        pending.status, pending.finished_at, pending.result = "failed", now(), {"error": "sync_timed_out"}
        # A late response from that attempt must never overwrite the next snapshot.
        account.generation += 1
    if not force and account.next_sync_at and account.next_sync_at > now():
        return None
    job = enqueue_job(db, owner, "google_sync", {"generation": account.generation})
    account.next_sync_at = now() + timedelta(seconds=get_settings().google_poll_seconds)
    return job.id


def select_calendar(db, owner, args):
    advisory(db, f"google:{owner}")
    account = db.get(GoogleIdentity, owner)
    if not account or not account.calendar_enabled:
        raise DomainError("GOOGLE_RECONNECT", "Connect Calendar in Settings first.", 409)
    row = owned(db, GoogleCalendar, args.calendar_id, owner, lock=True)
    check_revision(row, args.expected_revision)
    if args.selected and not row.available:
        raise DomainError("INVALID_ARGUMENT", "That calendar is no longer accessible.")
    if row.selected != args.selected:
        row.selected = args.selected
        row.revision += 1
        row.sync_token, row.last_sync_at = None, None
        db.execute(delete(GoogleCalendarEvent).where(GoogleCalendarEvent.calendar_id == row.id))
        account.generation += 1
        account.next_sync_at = now()
    emit(db, owner, "google.changed", owner)
    db.flush()
    return {"id": row.id, "selected": row.selected, "revision": row.revision}


def pages(client, path, params, limit=20):
    items, page, seen = [], None, set()
    for _ in range(limit):
        data = client.request(path, {**params, **({"pageToken": page} if page else {})})
        items.extend(data.get("items", []))
        page = data.get("nextPageToken")
        if not page:
            return items, data.get("nextSyncToken")
        if page in seen:
            break
        seen.add(page)
    raise SyncFailure("too_many_events")


def clean_event(event):
    # Store the fields needed to display and expand events, not guest lists or conference secrets.
    data = {
        key: event[key]
        for key in [
            "id",
            "etag",
            "eventType",
            "locked",
            "summary",
            "start",
            "end",
            "status",
            "transparency",
            "recurrence",
            "recurringEventId",
            "originalStartTime",
            "htmlLink",
            "location",
        ]
        if key in event
    }
    from .calendar_details import details

    data.update(details(event))
    data["has_guests"] = bool(event.get("attendees"))
    data["declined"] = any(
        a.get("self") and a.get("responseStatus") == "declined" for a in event.get("attendees", [])
    )
    return data


def process(job_id):
    with session_scope() as db:
        job = db.get(Job, job_id)
        if not job or job.status in {"succeeded", "cancelled", "failed"}:
            return
        account = db.get(GoogleIdentity, job.owner_id)
        if not account or not account.calendar_enabled or account.generation != job.payload["generation"]:
            job.status, job.finished_at = "cancelled", now()
            return
        owner, generation, encrypted = account.owner_id, account.generation, account.credentials
        snapshots = {
            row.provider_id: {
                "id": row.id,
                "selected": row.selected,
                "sync_token": row.sync_token if row.details_version >= 1 else None,
                "revision": row.revision,
            }
            for row in db.scalars(select(GoogleCalendar).where(GoogleCalendar.owner_id == owner))
        }
        job.status = "running"
    client = None
    try:
        client = CalendarClient(unseal(encrypted))
        remote, _ = pages(
            client, "users/me/calendarList", {"maxResults": 250, "showDeleted": "true"}, limit=10
        )
        updates = {}
        visible = {item["id"]: item for item in remote if not item.get("deleted")}
        for cid, item in visible.items():
            previous = snapshots.get(cid, {})
            if not previous.get("selected", item.get("primary", False)):
                continue
            sync_token = previous.get("sync_token")
            path = "calendars/" + quote(cid, safe="") + "/events"
            params = {"maxResults": 2500, "singleEvents": "false", "showDeleted": "true"}
            try:
                events, next_token = pages(
                    client, path, {**params, **({"syncToken": sync_token} if sync_token else {})}
                )
            except SyncFailure as error:
                if error.status == 410 and sync_token:
                    sync_token = None
                    events, next_token = pages(client, path, params)
                elif error.status in {403, 404}:
                    updates[cid] = {"unavailable": True}
                    continue
                else:
                    raise
            if not next_token:
                raise SyncFailure("missing_sync_cursor")
            updates[cid] = {
                "full": not sync_token,
                "events": [clean_event(e) for e in events],
                "token": next_token,
            }
        with session_scope() as db:
            advisory(db, f"google:{owner}")
            account, job = db.get(GoogleIdentity, owner), db.get(Job, job_id)
            if not account or not account.calendar_enabled or account.generation != generation:
                job.status, job.finished_at = "cancelled", now()
                return
            for row in db.scalars(select(GoogleCalendar).where(GoogleCalendar.owner_id == owner)):
                if row.provider_id not in visible:
                    row.available, row.sync_token, row.last_sync_at = False, None, None
                    row.revision += 1
                    db.execute(delete(GoogleCalendarEvent).where(GoogleCalendarEvent.calendar_id == row.id))
            for cid, item in visible.items():
                row = db.scalar(
                    select(GoogleCalendar).where(
                        GoogleCalendar.owner_id == owner, GoogleCalendar.provider_id == cid
                    )
                )
                if not row:
                    row = GoogleCalendar(
                        owner_id=owner,
                        provider_id=cid,
                        title=item.get("summary", "Calendar")[:500],
                        primary=bool(item.get("primary")),
                        selected=bool(item.get("primary")),
                    )
                    db.add(row)
                    db.flush()
                row.title, row.timezone = item.get("summary", "Calendar")[:500], item.get("timeZone", "UTC")
                row.access_role = item.get("accessRole", "reader")
                update = updates.get(cid)
                row.available = not (update and update.get("unavailable"))
                if update and update.get("unavailable"):
                    row.sync_token, row.last_sync_at = None, None
                    db.execute(delete(GoogleCalendarEvent).where(GoogleCalendarEvent.calendar_id == row.id))
                    continue
                if not update:
                    continue
                if cid in snapshots and row.revision != snapshots[cid]["revision"]:
                    # A local write completed after this sync fetched its snapshot.
                    account.next_sync_at = now()
                    continue
                if update["full"]:
                    incoming = {e["id"] for e in update["events"]}
                    db.execute(
                        delete(GoogleCalendarEvent).where(
                            GoogleCalendarEvent.calendar_id == row.id,
                            GoogleCalendarEvent.provider_id.notin_(incoming),
                        )
                    )
                current = {
                    event.provider_id: event
                    for event in db.scalars(
                        select(GoogleCalendarEvent).where(GoogleCalendarEvent.calendar_id == row.id)
                    )
                }
                for item in update["events"]:
                    event = current.get(item["id"])
                    if event:
                        event.payload = item
                    else:
                        event = GoogleCalendarEvent(calendar_id=row.id, provider_id=item["id"], payload=item)
                        current[item["id"]] = event
                        db.add(event)
                from .planning import reconcile

                reconcile(db, owner, row.id, update["events"], update["full"])
                row.sync_token, row.last_sync_at, row.details_version = update["token"], now(), 1
            account.status, account.error, account.last_sync_at = "ready", "", now()
            job.status, job.finished_at, job.result = "succeeded", now(), {"synced": True}
            emit(db, owner, "google.changed", owner)
    except (SyncFailure, DomainError, KeyError, ValueError, TypeError) as error:
        code = (
            error.code
            if isinstance(error, SyncFailure)
            else "reconnect"
            if isinstance(error, DomainError)
            else "invalid_response"
        )
        with session_scope() as db:
            advisory(db, f"google:{owner}")
            account, job = db.get(GoogleIdentity, owner), db.get(Job, job_id)
            job.status, job.finished_at, job.result = "failed", now(), {"error": code}
            if account and account.calendar_enabled and account.generation == generation:
                account.status = "needs_reconnect" if code == "reconnect" else "error"
                account.error = code
                account.next_sync_at = now() + timedelta(seconds=get_settings().google_poll_seconds)
                emit(db, owner, "google.changed", owner)
    finally:
        if client:
            client.close()


def availability(owner, start, end, minutes=30):
    try:
        begin, until = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if not begin.tzinfo or not until.tzinfo or not timedelta(0) < until - begin <= timedelta(days=7):
            raise ValueError()
        if not 5 <= minutes <= 480:
            raise ValueError()
    except (ValueError, TypeError):
        raise DomainError(
            "INVALID_ARGUMENT", "Choose a timezone-aware interval of up to seven days and 5–480 minutes."
        )
    with session_scope() as db:
        account = db.get(GoogleIdentity, owner)
        if not account or not account.calendar_enabled:
            from .planning import local_availability

            return local_availability(owner, begin, until, minutes)
        if (
            not configured()
            or not account
            or not account.calendar_enabled
            or account.status == "needs_reconnect"
        ):
            return {
                "status": "unavailable",
                "reason": "Connect Google Calendar in Settings.",
                "free": [],
                "busy": [],
            }
        encrypted, generation = account.credentials, account.generation
        calendars = list(
            db.scalars(
                select(GoogleCalendar).where(
                    GoogleCalendar.owner_id == owner, GoogleCalendar.selected.is_(True)
                )
            )
        )
        if not calendars or any(not row.available for row in calendars):
            return {
                "status": "unavailable",
                "reason": "Select accessible calendars and sync them in Settings.",
                "free": [],
                "busy": [],
            }
        identifiers = [row.provider_id for row in calendars]
    if len(identifiers) > 50:
        return {
            "status": "unavailable",
            "reason": "Select at most 50 calendars for availability.",
            "free": [],
            "busy": [],
        }
    client = None
    try:
        client = CalendarClient(unseal(encrypted))
        result = client.request(
            "freeBusy",
            body={
                "timeMin": begin.isoformat(),
                "timeMax": until.isoformat(),
                "items": [{"id": cid} for cid in identifiers],
            },
        )
        raw = result.get("calendars", {})
        if any(cid not in raw or raw[cid].get("errors") for cid in identifiers):
            raise SyncFailure("partial_availability")
        # Consent/selection may have changed while Google answered.
        with session_scope() as db:
            current = db.get(GoogleIdentity, owner)
            if not current or not current.calendar_enabled or current.generation != generation:
                raise SyncFailure("connection_changed")
        busy = []
        for cid in identifiers:
            for block in raw[cid].get("busy", []):
                a, b = datetime.fromisoformat(block["start"]), datetime.fromisoformat(block["end"])
                if not a.tzinfo or not b.tzinfo:
                    raise SyncFailure("invalid_availability")
                a, b = max(begin, a), min(until, b)
                if b > a:
                    busy.append((a, b))
        from .planning import busy_intervals

        busy.extend(busy_intervals(owner, begin, until))
        merged = []
        for a, b in sorted(busy):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        free, cursor = [], begin
        for a, b in merged + [(until, until)]:
            if a - cursor >= timedelta(minutes=minutes):
                free.append({"start": cursor.isoformat(), "end": a.isoformat()})
            cursor = max(cursor, b)
        return {
            "status": "fresh",
            "checked_at": now().isoformat(),
            "start": begin.isoformat(),
            "end": until.isoformat(),
            "free": free,
            "busy": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in merged],
            "calendar_count": len(identifiers),
            "source": "google_freebusy_and_eridani",
            "note": "Task deadlines and reminders are not reserved time blocks.",
        }
    except (SyncFailure, DomainError, ValueError, KeyError):
        return {
            "status": "unavailable",
            "reason": "Google could not confirm availability. Try again or reconnect in Settings.",
            "free": [],
            "busy": [],
        }
    finally:
        if client:
            client.close()


def event_detail(db, owner, event_id):
    row = db.execute(
        select(GoogleCalendarEvent, GoogleCalendar, GoogleIdentity)
        .join(GoogleCalendar, GoogleCalendar.id == GoogleCalendarEvent.calendar_id)
        .join(GoogleIdentity, GoogleIdentity.owner_id == GoogleCalendar.owner_id)
        .where(
            GoogleCalendarEvent.id == event_id,
            GoogleCalendar.owner_id == owner,
            GoogleCalendar.selected.is_(True),
            GoogleCalendar.available.is_(True),
            GoogleIdentity.calendar_enabled.is_(True),
        )
    ).first()
    if not row or row[0].payload.get("status") == "cancelled":
        raise DomainError("NOT_FOUND", "Calendar event is no longer available.", 404)
    return {
        **row[0].payload,
        "provider_id": row[0].provider_id,
        "title": row[0].payload.get("summary", "Untitled event"),
        "id": row[0].id,
        "calendar_id": row[1].id,
        "calendar_title": row[1].title,
        "last_sync_at": row[1].last_sync_at,
    }
