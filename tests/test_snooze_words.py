from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from jarvis.domain import DomainError
from jarvis.snooze_words import snooze_time

CHICAGO = "America/Chicago"
# Friday 2026-10-30, 10:00 in Chicago (CDT, UTC-5).
FRIDAY_10AM = datetime(2026, 10, 30, 15, tzinfo=UTC)


def local(words, at=FRIDAY_10AM, zone=CHICAGO, **kw):
    return snooze_time(words, zone, at, **kw).astimezone(ZoneInfo(zone))


@pytest.mark.parametrize(
    "words,delta",
    [
        ("in 2 hours", timedelta(hours=2)),
        ("2h", timedelta(hours=2)),
        ("1h30m", timedelta(minutes=90)),
        ("1 hour 30 minutes", timedelta(minutes=90)),
        ("in an hour and a half", timedelta(minutes=90)),
        ("half an hour", timedelta(minutes=30)),
        ("for 20m", timedelta(minutes=20)),
        ("in a couple of hours", timedelta(hours=2)),
        ("3 days", timedelta(days=3)),
        ("Snooze until in 45 mins.", timedelta(minutes=45)),
        ("later", timedelta(hours=3)),
    ],
)
def test_spans_add_real_elapsed_time(words, delta):
    assert snooze_time(words, CHICAGO, FRIDAY_10AM) == FRIDAY_10AM + delta


@pytest.mark.parametrize(
    "words,expected",
    [
        ("tomorrow", datetime(2026, 10, 31, 9)),
        ("tomorrow morning", datetime(2026, 10, 31, 9)),
        ("until tomorrow afternoon", datetime(2026, 10, 31, 14)),
        ("tomorrow at 9", datetime(2026, 10, 31, 9)),
        ("9am tomorrow", datetime(2026, 10, 31, 9)),
        ("tomorrow at 2:30", datetime(2026, 10, 31, 14, 30)),
        ("after lunch", datetime(2026, 10, 30, 13)),
        ("this evening", datetime(2026, 10, 30, 18)),
        ("tonight", datetime(2026, 10, 30, 20)),
        ("tonight at 11pm", datetime(2026, 10, 30, 23)),
        ("at 3", datetime(2026, 10, 30, 15)),
        ("15:30", datetime(2026, 10, 30, 15, 30)),
        ("02:30", datetime(2026, 10, 31, 2, 30)),
        ("end of day", datetime(2026, 10, 30, 17)),
        # A weekday is the next one, today included while its time is still ahead.
        ("friday 3pm", datetime(2026, 10, 30, 15)),
        ("friday", datetime(2026, 11, 6, 9)),
        ("on tuesday at 2:30pm", datetime(2026, 11, 3, 14, 30)),
        ("monday", datetime(2026, 11, 2, 9)),
        ("next week", datetime(2026, 11, 2, 9)),
        ("this weekend", datetime(2026, 10, 31, 9)),
    ],
)
def test_named_times_are_local_wall_clock(words, expected):
    assert local(words).replace(tzinfo=None) == expected


def test_a_named_time_without_a_day_means_the_next_one():
    three_pm = datetime(2026, 10, 30, 20, tzinfo=UTC)
    assert local("after lunch", three_pm).replace(tzinfo=None) == datetime(2026, 10, 31, 13)
    with pytest.raises(DomainError, match="already passed"):
        local("this morning")  # 'this' pins today; 9:00 is gone.


def test_after_work_follows_the_owners_hours():
    assert local("after work", work_end=time(18, 30)).replace(tzinfo=None) == datetime(2026, 10, 30, 18, 30)


def test_resolved_in_the_owners_time_zone():
    london = snooze_time("tomorrow morning", "Europe/London", FRIDAY_10AM)
    chicago = snooze_time("tomorrow morning", CHICAGO, FRIDAY_10AM)
    assert london == datetime(2026, 10, 31, 9, tzinfo=UTC)  # 09:00 GMT (the UK changed clocks a week earlier)
    assert chicago == datetime(2026, 10, 31, 14, tzinfo=UTC)  # 09:00 CDT


def test_spring_forward_moves_a_skipped_time_past_the_gap():
    # Chicago skips 02:00-03:00 on Sunday 2027-03-14. Saturday 14:00 CST:
    saturday = datetime(2027, 3, 13, 20, tzinfo=UTC)
    assert snooze_time("tomorrow at 2:30am", CHICAGO, saturday) == datetime(2027, 3, 14, 8, 30, tzinfo=UTC)
    # Named times keep the wall clock (09:00 CDT); spans keep elapsed time (15:00 CDT, not 14:00).
    assert snooze_time("tomorrow morning", CHICAGO, saturday) == datetime(2027, 3, 14, 14, tzinfo=UTC)
    assert local("in 24 hours", saturday).hour == 15


def test_fall_back_takes_the_first_of_a_repeated_time():
    # Chicago repeats 01:00-02:00 on Sunday 2026-11-01. Saturday 15:00 CDT:
    saturday = datetime(2026, 10, 31, 20, tzinfo=UTC)
    assert snooze_time("tomorrow at 1:30am", CHICAGO, saturday) == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    assert snooze_time("tomorrow morning", CHICAGO, saturday) == datetime(2026, 11, 1, 15, tzinfo=UTC)


def test_iso_still_works_and_nonsense_is_refused():
    assert snooze_time("2026-11-02T08:00", CHICAGO, FRIDAY_10AM) == datetime(2026, 11, 2, 14, tzinfo=UTC)
    for words in ("blah", "", "at 13pm", "tomorrow at 25:00"):
        with pytest.raises(DomainError) as error:
            snooze_time(words, CHICAGO, FRIDAY_10AM)
        assert error.value.code == "INVALID_ARGUMENT"


def test_every_snooze_choice_in_the_app_parses():
    import re
    from pathlib import Path

    source = (Path(__file__).parents[1] / "apps/web/src/NoticeSnooze.tsx").read_text()
    words = re.findall(r'until: "([^"]+)"', source)
    assert len(words) >= 3, "NoticeSnooze.tsx SNOOZE_CHOICES parser"
    # Early on a Monday, so every choice (including 'this evening') is still ahead.
    monday = datetime(2026, 11, 2, 13, tzinfo=UTC)
    for choice in words:
        assert snooze_time(choice, CHICAGO, monday) > monday, choice
