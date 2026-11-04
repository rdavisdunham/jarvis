"""NativeReconciler and core application: keyed replacement, LLM duplicate-or-contradiction check (mocked),
shadow mode, contested items, multi-valued kinds, durable negatives, stale ops and arrival-order convergence."""

from datetime import timedelta
from itertools import permutations

import pytest
from jarvis import cost_features, memory_learning
from jarvis.context import decisions, evidence, temporal
from jarvis.context.engine import Reconciler
from jarvis.context.models import ContextOwnerState
from jarvis.context.native import reconcile
from jarvis.context.native.reconcile import NativeReconciler
from jarvis.context.schema import ReconcileOp, ReconcileRequest
from jarvis.db import session_scope

from .test_temporal import OWNER, World, days, state

PROJECT = ("project", "p1")


class Model:
    """A mocked model: classifies pairs by a lookup of (item statement, candidate statement)."""

    def __init__(self, relations=None):
        self.relations, self.calls = relations or {}, []

    def __call__(self, owner, items):
        self.calls.append(items)
        return [{"item_id": i["item_id"], "candidate_id": c["candidate_id"],
                 "relation": self.relations.get((i["statement"], c["statement"]),
                                                self.relations.get((c["statement"], i["statement"]), "unrelated"))}
                for i in items for c in i["candidates"]]


def decision(w, text, offset, **kw):
    return w.assertion(text, days(offset), kind="decision", key=None, subject=PROJECT, **kw)


def test_native_reconciler_satisfies_the_protocol(history):
    engine = NativeReconciler(lambda owner, items: [])
    assert isinstance(engine, Reconciler)
    ops = engine.reconcile(ReconcileRequest(owner_id=OWNER, changed=history[1:], related=history[:1]))
    # Project-level due dates are not linked work: no deterministic replacement.
    assert {op.op for op in ops} == {"no_change"}


def test_keyed_due_date_is_replaced_deterministically_without_the_model():
    model = Model()
    with session_scope() as db:
        w = World(db)
        old, new = w.assertion("Due 2026-11-13.", days(-9)), w.assertion("Due 2026-11-20.", days(-2))
        w.reconcile([old, new], check=model)
        assert (old.lifecycle, new.lifecycle, new.supersedes_id) == ("superseded", "current", old.id)
        assert model.calls == []


def test_contradiction_with_clear_order_is_shadowed_by_default():
    model = Model({("Launch on Postgres.", "Launch on MySQL."): "contradiction"})
    with session_scope() as db:
        w = World(db)
        old = decision(w, "Launch on Postgres.", -9)
        w.reconcile([old], check=model)
        new = decision(w, "Launch on MySQL.", -2)
        summary = w.reconcile([new], check=model)
        assert summary["shadow"] == [[old.id, new.id]]
        assert (old.lifecycle, old.valid_to) == ("current", None)
        assert (new.lifecycle, new.supersedes_id) == ("shadow", old.id)
        assert len(model.calls) == 1


def test_owner_setting_enables_semantic_supersession():
    model = Model({("Launch on Postgres.", "Launch on MySQL."): "contradiction"})
    with session_scope() as db:
        db.add(ContextOwnerState(owner_id=OWNER, settings={"semantic_supersede": True}))
        w = World(db)
        old, new = decision(w, "Launch on Postgres.", -9), decision(w, "Launch on MySQL.", -2)
        w.reconcile([old, new], check=model)
        assert (old.lifecycle, old.valid_to, new.lifecycle) == ("superseded", days(-2), "current")


def test_unclear_order_is_contested_and_reaches_questions():
    model = Model({("Launch on Postgres.", "Launch on MySQL."): "contradiction"})
    with session_scope() as db:
        w = World(db)
        a, b = decision(w, "Launch on Postgres.", -2), decision(w, "Launch on MySQL.", -2)
        w.reconcile([a, b], check=model)
        assert (a.lifecycle, b.lifecycle) == ("contested", "contested")
        items = decisions.question_items(db, OWNER)
        assert len(items) == 1 and {c["assertion_id"] for c in items[0]["candidates"]} == {a.id, b.id}
        assert items[0]["kind"] == "context" and items[0]["answer_tool"] == "context.resolve"


def test_duplicates_merge_evidence_and_corroborate_only_across_sources():
    model = Model({("We launch on Postgres.", "Launch the pilot on Postgres."): "duplicate"})
    with session_scope() as db:
        w = World(db)
        source = w.source("We launch on Postgres.")
        first = decision(w, "We launch on Postgres.", -9, source=source)
        w.reconcile([first], check=model)
        again = decision(w, "Launch the pilot on Postgres.", -2)
        summary = w.reconcile([again], check=model)
        assert summary["merged"] == [[first.id, again.id]]
        assert (first.lifecycle, again.lifecycle) == ("current", "retracted")
        assert evidence.corroboration(db, OWNER, first.id) == 2
        repeat = decision(w, "We launch on Postgres.", -1, source=source)  # the same quote again
        w.reconcile([repeat], check=model)
        assert repeat.lifecycle == "retracted" and evidence.corroboration(db, OWNER, first.id) == 2
        derived = decision(w, "We launch on Postgres.", -1, origin="derived")
        w.reconcile([derived], check=model)
        assert evidence.corroboration(db, OWNER, first.id) == 2


def test_multi_valued_kinds_never_conflict_or_supersede():
    model = Model()
    with session_scope() as db:
        w = World(db)
        rows = [w.assertion("Priya works on the pilot.", days(-9), kind="relationship", key="works_on",
                            subject=("actor", "priya")),
                w.assertion("Priya works on billing.", days(-2), kind="relationship", key="works_on",
                            subject=("actor", "priya"))]
        w.reconcile(rows, check=model)
        assert [r.lifecycle for r in rows] == ["current", "current"] and model.calls == []
        ops = [ReconcileOp(op="supersede", targets=[rows[0].id, rows[1].id])]
        assert temporal.apply_ops(db, OWNER, ops, semantic_supersede=True)["refused"] == [[rows[0].id, rows[1].id]]
        assert rows[0].lifecycle == "current"


def test_retracted_interpretations_never_return_unchanged():
    with session_scope() as db:
        w = World(db)
        row = decision(w, "Launch on Postgres.", -9)
        decisions.wrong(db, OWNER, row.id, row.current_version, command_id="cmd-1", principal=OWNER)
        again = decision(w, "launch on postgres", -2)
        assert again.lifecycle == "retracted"
        assert temporal.history(db, OWNER, again.id)[0].decision_id is not None
        assert w.reconcile([again]) == {}
        different = decision(w, "Launch on Postgres in December.", -1)
        assert different.lifecycle == "current"


def test_stale_ops_and_unknown_model_ids_are_ignored():
    with session_scope() as db:
        w = World(db)
        a, b = decision(w, "Launch on Postgres.", -2), decision(w, "Launch on MySQL.", -2)
        op = ReconcileOp(op="conflict", targets=[a.id, b.id], target_revisions={a.id: 1, b.id: 7})
        assert temporal.apply_ops(db, OWNER, [op])["stale"] == [[a.id, b.id]]
        assert a.lifecycle == "current"

        def rogue(owner, items):
            return [{"item_id": items[0]["item_id"], "candidate_id": "not-offered", "relation": "contradiction"},
                    {"item_id": "nope", "candidate_id": a.id, "relation": "duplicate"}]

        w.reconcile([b], check=rogue)
        assert (a.lifecycle, b.lifecycle) == ("current", "current")


def test_free_text_arrival_order_converges():
    texts = [("We launch on Postgres.", -9), ("Launch the pilot on Postgres.", -5), ("Budget is 40k.", -3)]
    model = Model({("We launch on Postgres.", "Launch the pilot on Postgres."): "duplicate"})
    results = []
    for n, order in enumerate(permutations(range(len(texts)))):
        with session_scope() as db:
            w = World(db, f"free-{n}")
            for i in order:
                w.reconcile([decision(w, texts[i][0], texts[i][1])], check=model)
            results.append({k: v[0] for k, v in state(db, w.owner).items()})
    assert all(r == results[0] for r in results)
    assert results[0] == {"We launch on Postgres.": "current", "Launch the pilot on Postgres.": "retracted",
                          "Budget is 40k.": "current"}


def test_provider_check_uses_settings_budget_and_cost_label(monkeypatch):
    seen = {}

    def request(owner, path, payload, model, allowance, timeout):
        seen.update(owner=owner, path=path, allowance=allowance, timeout=timeout, payload=payload,
                    feature=cost_features.current_feature.get())
        return {"choices": [{"message": {"content": '{"pairs": [{"item_id": "a", "candidate_id": "b", '
                                                    '"relation": "duplicate"}]}'}}]}

    monkeypatch.setattr(memory_learning, "extraction_request", request)
    pairs = reconcile.provider_check(OWNER, [{"item_id": "a", "statement": "x", "candidates": []}])
    assert pairs == [{"item_id": "a", "candidate_id": "b", "relation": "duplicate"}]
    assert seen["feature"] == "context_reconciliation" and seen["path"] == "chat/completions"
    assert seen["timeout"] == 90 and 0 < seen["allowance"] <= 1
    assert seen["payload"]["response_format"]["json_schema"]["strict"] is True
    monkeypatch.setattr(memory_learning, "extraction_request",
                        lambda *a, **k: {"choices": [{"message": {"refusal": "no", "content": None}}]})
    assert reconcile.provider_check(OWNER, []) == []


@pytest.mark.parametrize("offset", [-2, 5])
def test_human_change_wins_over_keyed_chain_with_supported_time(offset):
    with session_scope() as db:
        w = World(db)
        old = w.assertion("Due 2026-11-13.", days(-9))
        w.reconcile([old])
        made = decisions.changed(db, OWNER, old.id, old.current_version, statement="Due 2026-11-27.",
                                 effective_at=days(offset), precision="day", command_id="cmd-change",
                                 principal=OWNER)
        new = db.get(temporal.ContextAssertion, made.after["successor_id"])
        assert new.supersedes_id == old.id and new.valid_from == days(offset)
        assert old.valid_to == days(offset)
        assert old.lifecycle == ("superseded" if offset < 0 else "current")
        assert days(offset) - days(-9) > timedelta(0)
