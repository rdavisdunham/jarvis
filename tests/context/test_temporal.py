"""Temporal model: recorded history, as-of filters, keyed chains, future successors, late arrivals, ties."""

from datetime import timedelta
from itertools import permutations

from jarvis.context import evidence, temporal
from jarvis.context.models import ContextAssertion, ContextSegment, ContextSource, ContextSourceRevision
from jarvis.context.native.reconcile import NativeReconciler
from jarvis.context.schema import CandidateAssertion, EntityRef, EvidenceRef, ScopePredicate, TimeBound
from jarvis.db import session_scope
from jarvis.models import now
from sqlalchemy import select

OWNER = "davin"
TASK = ("task", "task-1")


class World:
    """Sources with one segment per statement, and assertions admitted through `temporal.create`."""

    def __init__(self, db, owner=OWNER):
        self.db, self.owner, self.n = db, owner, 0

    def source(self, text, occurred_at=None):
        self.n += 1
        source = ContextSource(owner_id=self.owner, connector="api", external_id=f"m-{self.owner}-{self.n}",
                               current_revision=1)
        self.db.add(source)
        self.db.flush()
        revision = ContextSourceRevision(owner_id=self.owner, source_id=source.id, revision=1, content_hash="h" * 64,
                                         occurred_at=occurred_at or now())
        self.db.add(revision)
        self.db.flush()
        segment = ContextSegment(owner_id=self.owner, revision_id=revision.id, seq=0, text=text, char_start=0,
                                 char_end=len(text))
        self.db.add(segment)
        self.db.flush()
        return source, revision

    def assertion(self, statement, valid_from=None, *, precision="day", kind="commitment", subject=TASK,
                  key="due_date", source=None, origin="independent", work_key=None):
        valid_from = valid_from or now()
        if source is None:
            source = self.source(statement, valid_from)
        _, revision = source
        candidate = CandidateAssertion(
            kind=kind, statement=statement, key=key,
            subject=EntityRef(entity_type=subject[0], entity_id=subject[1]) if subject else None,
            valid_from=TimeBound(at=valid_from, precision=precision) if precision != "unknown" else None,
            responsible=[EntityRef(entity_type="actor", mention="Priya")] if kind == "commitment" else [],
            evidence=[EvidenceRef(segment_seq=0, quote=statement, char_start=0, char_end=len(statement),
                                  origin=origin)])
        row = temporal.create(self.db, self.owner, candidate, occurred_at=valid_from, work_key=work_key,
                              fallback_subject=("source", source[0].id))
        if precision == "unknown":
            row.valid_precision = "unknown"
        if row.lifecycle != "retracted":
            evidence.attach(self.db, self.owner, row, candidate.evidence, revision_id=revision.id)
        return row

    def reconcile(self, rows, check=None, **apply):
        request, work_keys = temporal.reconcile_request(self.db, self.owner, [r.id for r in rows])
        if request is None:
            return {}
        ops = NativeReconciler(check or (lambda owner, items: [])).reconcile(request, work_keys)
        return temporal.apply_ops(self.db, self.owner, ops, **apply)


def days(n):
    return now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=n)


def state(db, owner):
    rows = db.scalars(select(ContextAssertion).where(ContextAssertion.owner_id == owner))
    return {r.statement: (r.lifecycle, r.valid_from, r.valid_to) for r in rows}


def visible(db, scope):
    query = temporal.select_versions(scope) if scope.known_at else temporal.select_assertions(scope)
    return sorted(r.statement for r in db.scalars(query))


def test_versions_are_appended_and_known_at_reads_recorded_history():
    with session_scope() as db:
        w = World(db)
        row = w.assertion("Budget due 2026-11-13.", days(-5))
        first = now()
        temporal.record_version(db, row, statement="Budget due 2026-11-20.")
        assert temporal.record_version(db, row, statement="Budget due 2026-11-20.") is None  # no-op
        versions = temporal.history(db, OWNER, row.id)
        assert [v.version for v in versions] == [1, 2] and row.current_version == 2
        assert versions[0].recorded_to is not None and versions[1].recorded_to is None
        assert versions[0].statement == "Budget due 2026-11-13."
        scope = ScopePredicate(owner_id=OWNER, known_at=first)
        assert visible(db, scope) == ["Budget due 2026-11-13."]
        assert visible(db, ScopePredicate(owner_id=OWNER)) == ["Budget due 2026-11-20."]


def test_future_effective_successor_keeps_todays_value():
    with session_scope() as db:
        w = World(db)
        today = w.assertion("Due 2026-11-20.", days(-10))
        w.reconcile([today])
        future = w.assertion("Due 2026-12-01.", days(10))
        w.reconcile([future])
        assert (today.lifecycle, today.valid_to) == ("current", days(10))
        assert future.lifecycle == "current" and future.supersedes_id == today.id
        assert visible(db, ScopePredicate(owner_id=OWNER)) == ["Due 2026-11-20."]
        assert visible(db, ScopePredicate(owner_id=OWNER, valid_at=days(11))) == ["Due 2026-12-01."]
        assert visible(db, ScopePredicate(owner_id=OWNER, valid_at=days(-1))) == ["Due 2026-11-20."]
        assert temporal.settle(db, OWNER) == []
        assert temporal.settle(db, OWNER, at=days(12)) == [today.id]
        assert today.lifecycle == "superseded"


def test_late_arrival_splits_intervals_and_appends_history():
    with session_scope() as db:
        w = World(db)
        first, last = w.assertion("Due 2026-11-01.", days(-30)), w.assertion("Due 2026-11-20.", days(-10))
        w.reconcile([first, last])
        assert (first.lifecycle, first.valid_to) == ("superseded", days(-10))
        before_late = now()
        middle = w.assertion("Due 2026-11-10.", days(-20))
        w.reconcile([middle])
        assert first.valid_to == days(-20)
        assert (middle.lifecycle, middle.valid_to, middle.supersedes_id) == ("superseded", days(-10), first.id)
        assert (last.lifecycle, last.supersedes_id) == ("current", middle.id)
        assert [v.valid_to for v in temporal.history(db, OWNER, first.id)] == [None, days(-10), days(-20)]
        assert visible(db, ScopePredicate(owner_id=OWNER, valid_at=days(-15))) == ["Due 2026-11-10."]
        assert visible(db, ScopePredicate(owner_id=OWNER, valid_at=days(-15), known_at=before_late)) == [
            "Due 2026-11-01."]


def test_ties_and_unknown_times_never_invent_order():
    with session_scope() as db:
        w = World(db)
        a = w.assertion("Due 2026-11-20.", days(-3))
        b = w.assertion("Due 2026-11-27.", days(-3) + timedelta(hours=9), precision="instant")
        w.reconcile([a, b])
        assert (a.lifecycle, b.lifecycle) == ("contested", "contested")
        assert a.valid_to is None and b.valid_to is None

    with session_scope() as db:
        w = World(db, "other")
        known = w.assertion("Due 2026-11-20.", days(-3))
        unknown = w.assertion("Due 2026-12-24.", days(-1), precision="unknown")
        w.reconcile([known, unknown])
        assert (known.lifecycle, unknown.lifecycle) == ("contested", "contested")


def test_shuffled_arrival_orders_converge():
    statements = [("Due 2026-11-01.", -30, "day"), ("Due 2026-11-10.", -20, "day"), ("Due 2026-11-20.", -10, "day"),
                  ("Due 2026-11-21.", -10, "instant"), ("Due 2026-12-24.", 10, "day")]
    results = []
    for n, order in enumerate(permutations(range(len(statements)))):
        if n % 7:  # a spread of the 120 orders keeps the test fast
            continue
        with session_scope() as db:
            w = World(db, f"shuffle-{n}")
            for i in order:
                text, offset, precision = statements[i]
                w.reconcile([w.assertion(text, days(offset), precision=precision)])
            results.append(state(db, w.owner))
    assert len(results) > 10 and all(r == results[0] for r in results)
    final = results[0]
    assert final["Due 2026-11-20."][0] == final["Due 2026-11-21."][0] == "contested"
    assert final["Due 2026-12-24."][0] == "current"
    assert final["Due 2026-11-10."] == ("superseded", days(-20), days(-10))


def test_only_linked_work_is_single_valued():
    with session_scope() as db:
        w = World(db)
        rows = [w.assertion("Due 2026-11-13.", days(-9), subject=("project", "p1")),
                w.assertion("Due 2026-11-20.", days(-2), subject=("project", "p1"))]
        w.reconcile(rows)
        assert [r.lifecycle for r in rows] == ["current", "current"]
        keyed = [w.assertion("Send memo by 2026-11-13.", days(-9), subject=("project", "p2"), work_key="c-1"),
                 w.assertion("Send memo by 2026-11-20.", days(-2), subject=("project", "p2"), work_key="c-1")]
        w.reconcile(keyed)
        assert [r.lifecycle for r in keyed] == ["superseded", "current"]


def test_scope_filters_kinds_entities_and_never_shadow():
    with session_scope() as db:
        w = World(db)
        w.assertion("Due 2026-11-20.", days(-1))
        decision = w.assertion("Ship on Postgres.", days(-1), kind="decision", key=None, subject=("project", "p1"))
        hidden = w.assertion("Ship on MySQL.", days(-1), kind="decision", key=None, subject=("project", "p1"))
        temporal.record_version(db, hidden, lifecycle="shadow")
        project = EntityRef(entity_type="project", entity_id="p1")
        assert visible(db, ScopePredicate(owner_id=OWNER, entities=[project])) == [decision.statement]
        assert visible(db, ScopePredicate(owner_id=OWNER, kinds=["commitment"])) == ["Due 2026-11-20."]
        assert hidden.statement not in visible(db, ScopePredicate(owner_id=OWNER, include_superseded=True,
                                                                  include_retracted=True))
        assert visible(db, ScopePredicate(owner_id="someone-else")) == []


def test_precision_windows():
    start = days(0)
    assert temporal.precedes(temporal.span(start, "day"), temporal.span(start + timedelta(days=1), "instant"))
    assert not temporal.precedes(temporal.span(start, "day"), temporal.span(start + timedelta(hours=3), "instant"))
    assert not temporal.precedes(temporal.span(start, "unknown"), temporal.span(start + timedelta(days=99), "day"))
    month = temporal.span(start.replace(month=1, day=31), "month")
    assert month.end.month == 2 and month.end.day in (28, 29)
    assert temporal.span(start.replace(month=11, day=1), "quarter").end.year == start.year + 1
