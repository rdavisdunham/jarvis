"""Cached, safe-to-render Google event details and legacy recurrence normalization."""

import re
from datetime import UTC, date, datetime, time
from urllib.parse import urlsplit

from .domain import zone


def web_link(value):
    try:
        parsed = urlsplit(value or "")
        return (
            value
            if parsed.scheme in {"https", "http"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            else None
        )
    except ValueError:
        return None


def details(event):
    guests = []
    for guest in event.get("attendees", [])[:200]:
        guests.append(
            {
                k: guest[k]
                for k in ["displayName", "email", "responseStatus", "self", "optional", "organizer"]
                if k in guest
            }
        )
    meeting = web_link(event.get("hangoutLink"))
    if not meeting:
        meeting = next(
            (
                web_link(e.get("uri"))
                for e in event.get("conferenceData", {}).get("entryPoints", [])
                if e.get("entryPointType") == "video" and web_link(e.get("uri"))
            ),
            None,
        )
    return {
        "description": str(event.get("description") or "")[:20000],
        "meeting_url": meeting,
        "attendees": guests,
        "attendees_omitted": bool(event.get("attendeesOmitted")) or len(event.get("attendees", [])) > 200,
        "organizer": {
            k: event["organizer"][k]
            for k in ["displayName", "email", "self"]
            if k in event.get("organizer", {})
        },
        "attachments": [
            {"title": str(a.get("title") or "Attachment")[:500], "url": web_link(a.get("fileUrl"))}
            for a in event.get("attachments", [])[:25]
            if web_link(a.get("fileUrl"))
        ],
    }


def normalized_recurrence(lines, all_day, timezone):
    """Match UNTIL/EXDATE/RDATE value types to Google's start value, including legacy exports."""
    local = zone(timezone)

    def value(raw):
        if all_day:
            if len(raw) == 8:
                return raw
            parsed = (
                datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                if raw.endswith("Z")
                else datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(tzinfo=local)
            )
            return parsed.astimezone(local).strftime("%Y%m%d")
        if len(raw) == 8:
            return (
                datetime.combine(date.fromisoformat(raw), time(23, 59, 59), local)
                .astimezone(UTC)
                .strftime("%Y%m%dT%H%M%SZ")
            )
        if raw.endswith("Z"):
            return raw
        return (
            datetime.strptime(raw, "%Y%m%dT%H%M%S")
            .replace(tzinfo=local)
            .astimezone(UTC)
            .strftime("%Y%m%dT%H%M%SZ")
        )

    out = []
    for line in lines:
        if line.startswith("RRULE:"):
            line = re.sub(r"UNTIL=([^;]+)", lambda match: "UNTIL=" + value(match[1]), line)
        elif all_day and line.startswith(("EXDATE", "RDATE")):
            prefix, raw = line.split(":", 1)
            line = prefix.split(";")[0] + ":" + ",".join(value(part) for part in raw.split(","))
        out.append(line.replace(";VALUE=DATE:", ":"))
    return "\n".join(out)
