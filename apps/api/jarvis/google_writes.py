"""Durable, owner-authorized Google event writes with provider reconciliation."""

from datetime import date, datetime, timedelta
from urllib.parse import quote
from uuid import uuid4

from sqlalchemy import select

from . import google_calendar
from .db import session_scope
from .domain import DomainError, advisory, emit, enqueue_job, owned, parse_when, zone
from .google_auth import configured, seal, unseal
from .google_calendar import SyncFailure, clean_event
from .google_projection import original_key
from .models import GoogleCalendar, GoogleCalendarEvent, GoogleIdentity, Job, now

TERMINAL = {"succeeded", "failed", "cancelled", "unconfirmed"}
WRITE_ROLES = {"owner", "writer"}
MARKER = "eridaniWrite"


def account_source(db, owner, calendar_id, *, writing=False, generation=None):
    account = db.get(GoogleIdentity, owner)
    source = owned(db, GoogleCalendar, calendar_id, owner)
    if not configured() or not account or not account.calendar_enabled or account.status == "needs_reconnect":
        raise DomainError("GOOGLE_RECONNECT", "Reconnect Google Calendar in Settings.", 409)
    if generation is not None and account.generation != generation:
        raise DomainError("GOOGLE_CHANGED", "Google connection changed. Review the event and try again.", 409)
    if not source.selected or not source.available:
        raise DomainError("GOOGLE_CHANGED", "Select an accessible calendar in Settings first.", 409)
    if writing and (not account.calendar_write_enabled or source.access_role not in WRITE_ROLES):
        raise DomainError(
            "GOOGLE_READ_ONLY", "Enable Calendar editing in Settings and choose a writable calendar.", 403
        )
    return account, source


def blocked_event(event):
    if event.get("locked") or event.get("eventType", "default") != "default":
        return "This special event is managed in Google Calendar."
    if event.get("attendees") or event.get("organizer", {}).get("self") is False:
        return "Events with guests are managed in Google Calendar in this release."
    return ""


def event_body(args):
    location = zone(args.timezone)
    try:
        if args.all_day:
            begin, until = date.fromisoformat(args.start), date.fromisoformat(args.end)
            if not timedelta(0) < until - begin <= timedelta(days=366):
                raise ValueError()
            start, end = {"date": begin.isoformat()}, {"date": until.isoformat()}
        else:
            begin, until = parse_when(args.start, args.timezone), parse_when(args.end, args.timezone)
            if not timedelta(0) < until - begin <= timedelta(days=366):
                raise ValueError()
            for raw, parsed in [(args.start, begin), (args.end, until)]:
                local = datetime.fromisoformat(raw).replace(tzinfo=None)
                if parsed.astimezone(location).replace(tzinfo=None) != local:
                    raise ValueError()
            start = {"dateTime": begin.isoformat(), "timeZone": args.timezone}
            end = {"dateTime": until.isoformat(), "timeZone": args.timezone}
    except (ValueError, TypeError):
        raise DomainError(
            "INVALID_ARGUMENT",
            "Choose valid start/end times in the selected time zone, with end after start. All-day end dates are exclusive.",
        ) from None
    title = args.title.strip()
    if not title:
        raise DomainError("INVALID_ARGUMENT", "An event needs a title.")
    result = {
        "summary": title,
        "start": start,
        "end": end,
        "location": args.location,
        "description": args.description,
        "transparency": "opaque" if args.busy else "transparent",
    }
    if getattr(args, "repeat", "none") != "none":
        result["recurrence"] = ["RRULE:FREQ=" + args.repeat.upper()]
    return result


def event_path(calendar, event_id=None):
    return (
        "calendars/"
        + quote(calendar, safe="")
        + "/events"
        + ("/" + quote(event_id, safe="") if event_id else "")
    )


def read_event(owner, args):
    with session_scope() as db:
        record = db.get(GoogleCalendarEvent, args.event_id)
        if not record:
            raise DomainError("NOT_FOUND", "Event no longer available. Refresh the calendar.", 404)
        account, source = account_source(db, owner, record.calendar_id)
        encrypted, generation = account.credentials, account.generation
        source_id, provider_calendar, timezone = source.id, source.provider_id, source.timezone
        write_enabled = account.calendar_write_enabled and source.access_role in WRITE_ROLES
        provider_event, cached = record.provider_id, record.payload
        title = source.title
    client = None
    try:
        client = google_calendar.CalendarClient(unseal(encrypted))
        series_id = cached.get("recurringEventId") or (provider_event if cached.get("recurrence") else None)
        if series_id:
            if args.scope == "event":
                raise DomainError("CALENDAR_SCOPE", "Choose this occurrence or the entire series.", 409)
            if args.scope == "series":
                provider_event = series_id
            else:
                original = args.occurrence_start
                if not original:
                    raise DomainError(
                        "INVALID_ARGUMENT", "Choose the original occurrence start from calendar_list."
                    )
                data = client.request(
                    event_path(provider_calendar, series_id) + "/instances",
                    {"originalStart": original, "showDeleted": "true", "maxResults": 10},
                )
                match = []
                try:
                    wanted = original_key(
                        {"date": original} if len(original) == 10 else {"dateTime": original}, timezone
                    )
                    for item in data.get("items", []):
                        if (
                            item.get("originalStartTime")
                            and original_key(item["originalStartTime"], timezone) == wanted
                        ):
                            match.append(item)
                except (ValueError, KeyError):
                    raise DomainError(
                        "INVALID_ARGUMENT", "Choose a valid original occurrence start."
                    ) from None
                if len(match) != 1:
                    raise DomainError("NOT_FOUND", "That occurrence is no longer available.", 404)
                provider_event = match[0]["id"]
        elif args.scope != "event":
            raise DomainError("CALENDAR_SCOPE", "This is a single event, not a recurring series.", 409)
        remote = client.request(event_path(provider_calendar, provider_event))
        if remote.get("status") == "cancelled":
            raise DomainError("NOT_FOUND", "That event was deleted.", 404)
        reason = blocked_event(remote)
        editable = bool(write_enabled and remote.get("etag") and not reason)
        with session_scope() as db:
            account_source(db, owner, source_id, generation=generation)
        token = (
            seal(
                {
                    "purpose": "calendar_edit",
                    "owner": owner,
                    "calendar_id": source_id,
                    "generation": generation,
                    "provider_event": provider_event,
                    "etag": remote.get("etag"),
                    "scope": args.scope,
                    "expires_at": (now() + timedelta(minutes=30)).isoformat(),
                    "base_event": clean_event(remote),
                }
            )
            if editable
            else None
        )
        return {
            "event_id": args.event_id,
            "calendar_id": source_id,
            "calendar_title": title,
            "title": remote.get("summary") or "Busy",
            "start": remote["start"].get("date") or remote["start"].get("dateTime"),
            "end": remote["end"].get("date") or remote["end"].get("dateTime"),
            "all_day": "date" in remote["start"],
            "timezone": remote["start"].get("timeZone") or timezone,
            "location": remote.get("location", ""),
            "description": remote.get("description", ""),
            "busy": remote.get("transparency") != "transparent",
            "recurring": bool(series_id),
            "scope": args.scope,
            "editable": editable,
            "edit_token": token,
            "read_only_reason": reason
            or ("" if editable else "Enable Calendar editing in Settings to change this event."),
        }
    except SyncFailure as error:
        raise DomainError(
            "GOOGLE_UNAVAILABLE", "Google could not load this event. Try again or reconnect in Settings.", 502
        ) from error
    finally:
        if client:
            client.close()


def queue_write(db, owner, tool, args):
    advisory(db, f"google:{owner}")
    operation = tool.split(".")[1]
    if operation == "create":
        account, source = account_source(db, owner, args.calendar_id, writing=True)
        payload = {
            "operation": operation,
            "calendar_id": source.id,
            "generation": account.generation,
            "provider_event": uuid4().hex,
            "body": event_body(args),
        }
    else:
        token = unseal(args.edit_token)
        try:
            valid = (
                token["purpose"] == "calendar_edit"
                and token["owner"] == owner
                and datetime.fromisoformat(token["expires_at"]) > now()
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise DomainError("CALENDAR_STALE", "Reopen the event to review its current details.", 409)
        account, source = account_source(
            db, owner, token["calendar_id"], writing=True, generation=token["generation"]
        )
        payload = {
            "operation": operation,
            "calendar_id": source.id,
            "generation": account.generation,
            "provider_event": token["provider_event"],
            "etag": token["etag"],
            "scope": token["scope"],
            "base_event": token["base_event"],
        }
        if operation == "update":
            payload["body"] = event_body(args)
    payload["provider_calendar"] = source.provider_id
    job = enqueue_job(db, owner, "google_write", payload)
    emit(db, owner, "google.write", job.id)
    return {
        "job_id": job.id,
        "status": "queued",
        "message": "Queued for Google. Check calendar_write_status before claiming it saved.",
    }


def write_status(db, owner, job_id):
    job = owned(db, Job, job_id, owner)
    if job.kind != "google_write":
        raise DomainError("NOT_FOUND", "Calendar change not found.", 404)
    return {
        "job_id": job.id,
        "status": job.status,
        "operation": job.payload["operation"],
        "created_at": job.created_at.isoformat(),
        "result": job.result,
    }


def completed(job_id, remote):
    with session_scope() as db:
        job = db.get(Job, job_id)
        payload, owner = job.payload, job.owner_id
        advisory(db, f"google:{owner}")
        account, source = db.get(GoogleIdentity, owner), db.get(GoogleCalendar, payload["calendar_id"])
        local_id = None
        if (
            account
            and source
            and account.calendar_enabled
            and account.generation == payload["generation"]
            and source.selected
        ):
            row = db.scalar(
                select(GoogleCalendarEvent).where(
                    GoogleCalendarEvent.calendar_id == source.id,
                    GoogleCalendarEvent.provider_id == payload["provider_event"],
                )
            )
            if not row:
                row = GoogleCalendarEvent(
                    calendar_id=source.id, provider_id=payload["provider_event"], payload=clean_event(remote)
                )
                db.add(row)
            else:
                row.payload = clean_event(remote)
            db.flush()
            local_id = row.id
            source.revision += 1  # Reject sync snapshots taken before this write.
            account.next_sync_at = now()
            emit(db, owner, "google.changed", owner)
        job.status, job.finished_at = "succeeded", now()
        job.result = {
            "operation": payload["operation"],
            "event_id": local_id,
            "title": remote.get("summary", ""),
            "deleted": payload["operation"] == "delete",
            "message": "Deleted from Google Calendar."
            if payload["operation"] == "delete"
            else "Saved in Google Calendar.",
        }
        emit(db, owner, "google.write", job.id)


def process_write(job_id):
    with session_scope() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if not job or job.status in TERMINAL:
            return
        attempts = (job.result or {}).get("attempts", 0) + 1
        started = (job.result or {}).get("write_started", False)
        previous_outcome_unknown = started
        owner, payload = job.owner_id, job.payload
        try:
            account, _ = account_source(
                db, owner, payload["calendar_id"], writing=True, generation=payload["generation"]
            )
            encrypted = account.credentials
        except DomainError:
            job.status, job.finished_at, job.result = (
                ("unconfirmed" if started else "cancelled"),
                now(),
                {"message": "Connection changed. Check Google Calendar for any change already in progress."},
            )
            return
        job.status, job.result = "running", {"attempts": attempts, "write_started": started}
    client = None
    try:
        client = google_calendar.CalendarClient(unseal(encrypted))
        acl = client.request("users/me/calendarList/" + quote(payload["provider_calendar"], safe=""))
        if acl.get("accessRole") not in WRITE_ROLES:
            raise DomainError("GOOGLE_READ_ONLY", "Google no longer allows editing this calendar.", 403)
        path = event_path(payload["provider_calendar"], payload["provider_event"])
        try:
            remote = client.request(path)
        except SyncFailure as error:
            if error.status not in {404, 410}:
                raise
            remote = None
        marker = (remote or {}).get("extendedProperties", {}).get("private", {}).get(MARKER)
        operation = payload["operation"]
        if operation == "delete" and (not remote or remote.get("status") == "cancelled"):
            completed(
                job_id,
                {**payload.get("base_event", {}), "id": payload["provider_event"], "status": "cancelled"},
            )
            return
        if operation != "delete" and remote and marker == job_id:
            completed(job_id, remote)
            return
        if operation == "create" and remote:
            raise DomainError(
                "CALENDAR_CONFLICT",
                "An event already occupies this identifier. No event was overwritten.",
                409,
            )
        if operation != "create":
            if not remote or remote.get("status") == "cancelled":
                raise DomainError("CALENDAR_CONFLICT", "This event was deleted. Refresh the calendar.", 409)
            if remote.get("etag") != payload["etag"]:
                raise DomainError(
                    "CALENDAR_CONFLICT", "This event changed in Google. Reopen it before editing.", 409
                )
            reason = blocked_event(remote)
            if reason:
                raise DomainError("CALENDAR_READ_ONLY", reason, 403)
        with session_scope() as db:
            account_source(db, owner, payload["calendar_id"], writing=True, generation=payload["generation"])
            job = db.get(Job, job_id)
            if now() - job.created_at > timedelta(hours=1):
                raise DomainError(
                    "CALENDAR_EXPIRED", "This queued change expired. Check Google Calendar and submit again."
                )
            job.result = {"attempts": attempts, "write_started": True}
        started = True
        if operation == "delete":
            client.request(
                path, {"sendUpdates": "none"}, method="DELETE", headers={"If-Match": payload["etag"]}
            )
            saved = {**remote, "status": "cancelled"}
        else:
            body = dict(payload["body"])
            private = dict((remote or {}).get("extendedProperties", {}).get("private", {}))
            private[MARKER] = job_id
            body["extendedProperties"] = {"private": private}
            if operation == "create":
                body["id"] = payload["provider_event"]
                saved = client.request(
                    event_path(payload["provider_calendar"]), {"sendUpdates": "none"}, body, method="POST"
                )
            else:
                saved = client.request(
                    path, {"sendUpdates": "none"}, body, method="PATCH", headers={"If-Match": payload["etag"]}
                )
        if saved.get("id") != payload["provider_event"] or (operation != "delete" and not saved.get("etag")):
            raise SyncFailure("invalid_write_response")
        completed(job_id, saved)
    except (SyncFailure, DomainError, KeyError, ValueError, TypeError) as error:
        if not isinstance(error, (SyncFailure, DomainError)):
            error = SyncFailure("invalid_write_response")
        retry = (
            isinstance(error, SyncFailure)
            and (error.status in {0, 409, 429} or error.status >= 500)
            and attempts < 5
        )
        uncertain = (
            previous_outcome_unknown
            or started
            and isinstance(error, SyncFailure)
            and (error.status == 0 or error.status >= 500)
        )
        message = (
            error.message
            if isinstance(error, DomainError)
            else "This event changed in Google. Reopen it before editing."
            if error.status == 412
            else "Reconnect Calendar editing in Settings."
            if error.status in {401, 403}
            else "Google could not confirm this change. Check its status before submitting another change."
        )
        with session_scope() as db:
            job = db.get(Job, job_id)
            job.status = "retrying" if retry else "unconfirmed" if uncertain else "failed"
            job.finished_at = None if retry else now()
            job.result = {"attempts": attempts, "write_started": started, "message": message}
            emit(db, owner, "google.write", job.id)
        if retry:
            raise RuntimeError("Google Calendar write will retry with the same event identity.") from None
    finally:
        if client:
            client.close()
