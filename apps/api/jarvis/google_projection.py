"""Expand read-only Google series and overrides for the app's bounded calendar window."""

from datetime import UTC, date, datetime, time, timedelta
from urllib.parse import urlsplit

from dateutil.rrule import rrulestr
from sqlalchemy import select

from .calendar_details import normalized_recurrence
from .domain import DomainError, zone
from .models import GoogleCalendar, GoogleCalendarEvent, GoogleIdentity


def instant(value, timezone):
    if "date" in value:
        return datetime.combine(date.fromisoformat(value["date"]), time.min, zone(timezone)), True
    parsed = datetime.fromisoformat(value["dateTime"])
    local = zone(value.get("timeZone") or timezone)
    return parsed.astimezone(local) if parsed.tzinfo else parsed.replace(tzinfo=local), False


def safe_link(value):
    try:
        parsed = urlsplit(value or "")
        return (
            value
            if parsed.scheme == "https"
            and parsed.hostname in {"calendar.google.com", "www.google.com"}
            and not parsed.username
            and not parsed.password
            else None
        )
    except ValueError:
        return None


def original_key(value, timezone):
    point, all_day = instant(value, timezone)
    return point.date().isoformat() if all_day else point.astimezone(UTC).isoformat()


def project(db, owner, start, end, timezone, warnings=None):
    account = db.get(GoogleIdentity, owner)
    if not account or not account.calendar_enabled:
        return [], False
    local = zone(timezone)
    begin, until = datetime.combine(start, time.min, local), datetime.combine(end, time.min, local)
    output, incomplete = [], False

    def warning(source, reason):
        if warnings is not None:
            item = {"calendar": source.title, "reason": reason}
            if item not in warnings:
                warnings.append(item)

    sources = db.scalars(
        select(GoogleCalendar).where(
            GoogleCalendar.owner_id == owner,
            GoogleCalendar.selected.is_(True),
            GoogleCalendar.available.is_(True),
        )
    )
    for source in sources:
        records = list(
            db.scalars(select(GoogleCalendarEvent).where(GoogleCalendarEvent.calendar_id == source.id))
        )
        cancelled_series = {
            row.provider_id
            for row in records
            if row.payload.get("status") == "cancelled" and not row.payload.get("recurringEventId")
        }
        overrides = {}
        for row in records:
            item = row.payload
            if item.get("recurringEventId") and item.get("originalStartTime"):
                try:
                    overrides[
                        (item["recurringEventId"], original_key(item["originalStartTime"], source.timezone))
                    ] = row
                except (ValueError, KeyError):
                    incomplete = True

        def append(row, point, finish, all_day, *, source=source):
            nonlocal incomplete
            item = row.payload
            if item.get("status") == "cancelled" or item.get("declined"):
                return
            # All-day dates stay dates in the viewer's calendar.
            if all_day:
                point = datetime.combine(point.date(), time.min, local)
                finish = datetime.combine(finish.date(), time.min, local)
            if finish <= begin or point >= until or finish <= point:
                return
            day = max(point.astimezone(local).date(), start)
            last_day = min(
                (finish.astimezone(local) - timedelta(microseconds=1)).date(), end - timedelta(days=1)
            )
            while day <= last_day:
                if len(output) >= 2000:
                    incomplete = True
                    return
                output.append(
                    {
                        "id": "google:" + row.id + ":" + point.isoformat() + ":" + day.isoformat(),
                        "entity_id": row.id,
                        "provider_id": row.provider_id,
                        "kind": "google",
                        "title": item.get("summary") or "Busy",
                        "date": day.isoformat(),
                        "at": None if all_day else point.isoformat(),
                        "end_at": finish.isoformat(),
                        "busy_start": point.isoformat(),
                        "all_day": all_day,
                        "status": "active",
                        "project_id": None,
                        "task_id": None,
                        "revision": 1,
                        "projected": bool(item.get("recurrence")),
                        "notification_id": None,
                        "calendar_id": source.id,
                        "calendar_title": source.title,
                        "url": safe_link(item.get("htmlLink")),
                        "location": item.get("location", ""),
                        "description": item.get("description", ""),
                        "meeting_url": item.get("meeting_url"),
                        "busy": item.get("transparency") != "transparent",
                        "read_only": not (
                            account.calendar_write_enabled and source.access_role in {"owner", "writer"}
                        ),
                        "recurring": bool(item.get("recurrence") or item.get("recurringEventId")),
                        "occurrence_start": (
                            item.get("originalStartTime", {}).get("date")
                            or item.get("originalStartTime", {}).get("dateTime")
                            or (point.date().isoformat() if all_day else point.isoformat())
                        )
                        if item.get("recurrence") or item.get("recurringEventId")
                        else None,
                    }
                )
                day += timedelta(days=1)

        for row in records:
            item = row.payload
            if item.get("status") == "cancelled" or item.get("recurringEventId"):
                continue
            try:
                point, all_day = instant(item["start"], source.timezone)
                finish, _ = instant(item["end"], source.timezone)
                duration = finish - point
                if duration <= timedelta(0):
                    continue
                if not item.get("recurrence"):
                    append(row, point, finish, all_day)
                    continue
                rules = normalized_recurrence(item["recurrence"], all_day, source.timezone)
                if any(freq in rules for freq in ["FREQ=SECONDLY", "FREQ=MINUTELY", "FREQ=HOURLY"]):
                    incomplete = True
                    warning(source, "A recurrence frequency is not supported in this view.")
                    continue
                series_start = point.replace(tzinfo=None) if all_day else point
                lower = begin.replace(tzinfo=None) if all_day else begin
                upper = until.replace(tzinfo=None) if all_day else until
                series = rrulestr(rules.replace(";VALUE=DATE:", ":"), dtstart=series_start, forceset=True)
                # xafter bounds output and avoids materializing unbounded recurrences.
                for count, occurrence in enumerate(series.xafter(lower - duration, inc=True), start=1):
                    if occurrence >= upper:
                        break
                    if count > 2000:
                        incomplete = True
                        break
                    key = occurrence.date().isoformat() if all_day else occurrence.astimezone(UTC).isoformat()
                    if (row.provider_id, key) not in overrides:
                        append(row, occurrence, occurrence + duration, all_day)
            except (DomainError, ValueError, KeyError, TypeError, OverflowError):
                incomplete = True
                warning(source, "An event has dates or recurrence data that could not be read.")
        # Moved instances can enter this window even if their original slot was outside it.
        for row in overrides.values():
            if (
                row.payload.get("status") == "cancelled"
                or row.payload.get("recurringEventId") in cancelled_series
            ):
                continue
            try:
                point, all_day = instant(row.payload["start"], source.timezone)
                finish, _ = instant(row.payload["end"], source.timezone)
                append(row, point, finish, all_day)
            except (ValueError, KeyError, TypeError):
                incomplete = True
                warning(source, "A moved occurrence has unreadable dates.")
    return output, incomplete
