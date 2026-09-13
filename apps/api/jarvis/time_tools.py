"""Resolve local clock times without creating or changing records."""

from datetime import UTC, datetime, timedelta

from dateutil import tz

from .domain import DomainError, zone


def resolve_time(local, timezone):
    location = zone(timezone)
    try:
        parsed = datetime.fromisoformat(local)
    except (ValueError, TypeError):
        raise DomainError(
            "INVALID_ARGUMENT", "Use an ISO local date and time, for example 2030-03-10T02:30."
        ) from None
    if "T" not in local and " " not in local:
        raise DomainError("INVALID_ARGUMENT", "Include both a date and a clock time.")

    def candidate(value):
        return {
            "local": value.isoformat(),
            "utc": value.astimezone(UTC).isoformat(),
            "offset": value.strftime("%z")[:3] + ":" + value.strftime("%z")[3:],
            "timezone": timezone,
        }

    wall = parsed.replace(tzinfo=None)
    aware = wall.replace(tzinfo=location)
    if parsed.tzinfo is not None:
        actual = parsed.astimezone(location)
        if actual.replace(tzinfo=None) != wall:
            return {
                "status": "offset_mismatch",
                "valid": False,
                "timezone": timezone,
                "message": "The supplied offset does not match this local clock time in the requested zone.",
                "resolved_in_zone": candidate(actual),
            }
        return {"status": "valid", "valid": True, "candidates": [candidate(actual)]}
    if not tz.datetime_exists(aware):
        adjacent = []
        # Covers non-hour transitions and a skipped calendar day; never suggests another gap.
        for direction in (-1, 1):
            for minutes in range(1, 2881):
                probe = (wall + timedelta(minutes=direction * minutes)).replace(tzinfo=location)
                if tz.datetime_exists(probe):
                    adjacent.append(candidate(probe))
                    break
        return {
            "status": "nonexistent",
            "valid": False,
            "timezone": timezone,
            "alternatives": adjacent,
            "message": "This local time does not exist. Ask the owner to choose a valid time before saving.",
        }
    if tz.datetime_ambiguous(aware):
        return {
            "status": "ambiguous",
            "valid": False,
            "timezone": timezone,
            "candidates": [candidate(aware.replace(fold=fold)) for fold in (0, 1)],
            "message": "This clock time occurs twice. Ask which offset the owner means before saving.",
        }
    return {"status": "valid", "valid": True, "candidates": [candidate(aware)]}
