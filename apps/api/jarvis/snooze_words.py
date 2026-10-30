"""Snooze times in words ("in 2 hours", "tomorrow morning", "after lunch"), in the owner's time zone.

Spans add real elapsed time, so "in 2 hours" is two hours even across a clock change. Named times are
local wall-clock times: a time the clocks skip moves forward past the gap, and a time that happens
twice resolves to its first occurrence. A named time without a day ("after lunch", "3pm") means the
next one; a weekday ("friday 3pm") means the next one too, today included; "today", "tonight",
"this evening" and "tomorrow" never move, so a time already past is refused.
"""

import re
from datetime import UTC, datetime, time, timedelta

from dateutil import tz

from .domain import DomainError, parse_when, zone

MORNING, LUNCH, AFTER_LUNCH, AFTERNOON, EVENING, TONIGHT, NIGHT = (
    time(9), time(12), time(13), time(14), time(18), time(20), time(21)
)
LATER = timedelta(hours=3)
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SHORT_DAYS = {d[:3]: i for i, d in enumerate(WEEKDAYS)} | {"tues": 1, "thur": 3, "thurs": 3, "weekend": 5}
UNITS = {
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), timedelta(minutes=1)),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), timedelta(hours=1)),
    **dict.fromkeys(("d", "day", "days"), timedelta(days=1)),
    **dict.fromkeys(("w", "wk", "wks", "week", "weeks"), timedelta(weeks=1)),
}
COUNTS = {"a": 1, "an": 1, "one": 1, "a couple of": 2, "a couple": 2, "couple of": 2, "two": 2,
          "a few": 3, "few": 3, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15,
          "twenty": 20, "thirty": 30, "forty five": 45, "forty-five": 45}
SPAN = re.compile(
    r"(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, COUNTS), key=len, reverse=True)) + r")\s*("
    + "|".join(sorted(UNITS, key=len, reverse=True)) + r")(?![a-z])"
)
CLOCK = re.compile(r"^(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?$")
HELP = "Try 'in 2 hours', 'tomorrow morning', 'after lunch', 'friday 3pm' or a date and time."


def unreadable():
    return DomainError("INVALID_ARGUMENT", "I couldn't read that snooze time. " + HELP)


def wall(day, clock, location):
    """A local date and clock time as UTC: skipped times move past the gap, repeated ones take the first."""
    value = datetime.combine(day, clock, tzinfo=location)
    if not tz.datetime_exists(value):
        value = tz.resolve_imaginary(value)
    return value.astimezone(UTC)


def span(text):
    """'2 hours', '1h30m', 'an hour and a half', 'half an hour' → timedelta, or None."""
    text = re.sub(r"^(?:in|for|after)\s+", "", text)
    if re.fullmatch(r"(?:a\s+)?half\s+(?:an\s+)?hour", text):
        return timedelta(minutes=30)
    half = text.endswith(" and a half")
    text = text.removesuffix(" and a half")
    total, rest = timedelta(), text
    for match in SPAN.finditer(text):
        count = COUNTS.get(match[1]) or float(match[1])
        total += count * UNITS[match[2]]
        rest = rest.replace(match[0], "", 1)
    if not total or re.sub(r"[\s,]|and", "", rest):
        return None
    if half:
        last = SPAN.findall(text)[-1][1]
        total += UNITS[last] / 2
    return total


def clock_time(text, work_end):
    """The clock part: 'morning', 'after lunch', '3pm', 'at 15:30', 'end of day' → time, or None."""
    text = re.sub(r"^(?:in the|this|at|around)\s+", "", text)
    named = {
        "morning": MORNING, "lunch": LUNCH, "lunchtime": LUNCH, "noon": LUNCH, "midday": LUNCH,
        "after lunch": AFTER_LUNCH, "afternoon": AFTERNOON, "evening": EVENING, "tonight": TONIGHT,
        "night": NIGHT, "end of day": work_end, "end of the day": work_end, "eod": work_end,
        "after work": work_end, "end of work": work_end,
    }
    if text in named:
        return named[text]
    match = CLOCK.match(text)
    if not match:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2] or 0), (match[3] or "").replace(".", "")
    if minute > 59 or hour > 23 or (meridiem and not 1 <= hour <= 12):
        return None
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif not meridiem and not match[1].startswith("0") and 1 <= hour <= 6:
        hour += 12  # "at 3" and "2:30" mean the afternoon; "at 9" and "02:30" mean what they say.
    return time(hour, minute)


def day_part(text, today):
    """The day part → (date, default clock, days to roll if that time has passed), or None.

    'today', 'tonight', 'tomorrow' never roll; 'friday' is today if still ahead, else next week."""
    text = text.removeprefix("on ").strip()
    if text in {"today", "this"}:
        return today, None, 0
    if text == "tonight":
        return today, TONIGHT, 0
    if text in {"tomorrow", "tmrw", "tmr"}:
        return today + timedelta(days=1), MORNING, 0
    if text == "next week":
        return today + timedelta(days=7 - today.weekday()), MORNING, 0
    name = re.sub(r"^(?:next|this|coming|the)\s+", "", text)
    index = WEEKDAYS.index(name) if name in WEEKDAYS else SHORT_DAYS.get(name)
    if index is None:
        return None
    return today + timedelta(days=(index - today.weekday()) % 7), MORNING, 7


def named_time(text, today, work_end):
    """Split words into a day part and a clock part, either order → (date, clock, roll days), or None.

    Without a day the next such clock time is meant ('after lunch' at 3pm is tomorrow's)."""
    tokens, undated = text.split(), None
    for cut in range(len(tokens) + 1):
        before, after = " ".join(tokens[:cut]), " ".join(tokens[cut:])
        for day_text, clock_text in ((before, after), (after, before)):
            found = day_part(day_text, today) if day_text else None
            clock = clock_time(clock_text, work_end) if clock_text else None
            if (day_text and found is None) or (clock_text and clock is None):
                continue
            if found and (clock or found[1]):
                return found[0], clock or found[1], found[2]
            if not day_text and clock and not undated:
                undated = today, clock, 1
    return undated


def snooze_time(words, timezone, instant=None, work_end=time(17)):
    """Resolve snooze words to a UTC instant, or raise INVALID_ARGUMENT."""
    instant = instant or datetime.now(UTC)
    text = " ".join(words.casefold().replace(",", " ").split()).strip(" .!")
    text = re.sub(r"^(?:snooze\s+)?(?:until|till|til|to|for)\s+", "", text)
    if not text:
        raise unreadable()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return parse_when(words.strip(), timezone)
    if text in {"later", "later today", "in a bit", "in a while"}:
        return instant + LATER
    if (delta := span(text)) is not None:
        return instant + delta
    location = zone(timezone)
    today = instant.astimezone(location).date()
    named = named_time(text, today, work_end)
    if not named:
        raise unreadable()
    day, clock, roll = named
    resolved = wall(day, clock, location)
    if resolved <= instant and roll:
        resolved = wall(day + timedelta(days=roll), clock, location)
    if resolved <= instant:
        raise DomainError("INVALID_ARGUMENT", "That time has already passed. " + HELP)
    return resolved
