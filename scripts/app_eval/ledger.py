"""Durable, process-safe campaign allowance; uncertain calls retain reservations."""

import json
import sqlite3
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from uuid import uuid4

from .spending import EvalLimit

SCALE = Decimal(1_000_000_000)


def units(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("Invalid dollar amount")
    return int((number * SCALE).to_integral_value(rounding=ROUND_CEILING))


def dollars(value):
    return str(Decimal(value) / SCALE)


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def initialize(self, cap=10, judge_cap=1, request_cap=10000):
        cap, judge_cap = units(cap), units(judge_cap)
        if not 0 < cap <= units(10) or not 0 <= judge_cap <= cap or request_cap < 1:
            raise ValueError("Campaign cap must be > $0 and <= $10; judge cap must fit inside it")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, cap INTEGER, judge_cap INTEGER, request_cap INTEGER, stopped TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, trial TEXT, kind TEXT, model TEXT, bound INTEGER, charged INTEGER, actual INTEGER, state TEXT, metadata TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            existing = db.execute("SELECT * FROM config WHERE id=1").fetchone()
            if existing and (existing["cap"], existing["judge_cap"], existing["request_cap"]) != (
                cap,
                judge_cap,
                request_cap,
            ):
                raise ValueError("Resume cannot change the original allowance")
            db.execute("INSERT OR IGNORE INTO config VALUES (1,?,?,?,NULL)", (cap, judge_cap, request_cap))

    def reserve(self, trial, kind, model, bound, metadata=None):
        amount = units(bound)
        if amount <= 0 or kind not in {"agent", "pipeline", "embedding", "judge", "voice"}:
            raise ValueError("A paid call needs a positive bound and known kind")
        with self.transaction() as db:
            config = db.execute("SELECT * FROM config WHERE id=1").fetchone()
            if config["stopped"] or (self.path.parent / "STOP").exists():
                raise EvalLimit("Campaign stopped")
            total, count = db.execute("SELECT COALESCE(SUM(charged),0), COUNT(*) FROM calls").fetchone()
            judge = db.execute("SELECT COALESCE(SUM(charged),0) FROM calls WHERE kind='judge'").fetchone()[0]
            if total + amount > config["cap"] or count >= config["request_cap"]:
                raise EvalLimit("Campaign dollar/request ceiling reached")
            if kind == "judge" and judge + amount > config["judge_cap"]:
                raise EvalLimit("Semantic judge sublimit reached")
            identity = uuid4().hex
            db.execute(
                "INSERT INTO calls (id,trial,kind,model,bound,charged,state,metadata) VALUES (?,?,?,?,?,?,'uncertain',?)",
                (identity, trial, kind, model, amount, amount, json.dumps(metadata or {})),
            )
        return identity

    def settle(self, identity, cost, metadata=None):
        actual = units(cost)
        exceeded = False
        with self.transaction() as db:
            row = db.execute("SELECT * FROM calls WHERE id=?", (identity,)).fetchone()
            if row is None:
                raise ValueError("Unknown reservation")
            if row["state"] == "recorded":
                if row["actual"] != actual:
                    raise ValueError("Conflicting duplicate settlement")
                return
            details = {**json.loads(row["metadata"]), **(metadata or {})}
            db.execute(
                "UPDATE calls SET charged=?,actual=?,state='recorded',metadata=? WHERE id=?",
                (actual, actual, json.dumps(details), identity),
            )
            exceeded = actual > row["bound"]
            if exceeded:
                db.execute("UPDATE config SET stopped='Provider usage exceeded reservation' WHERE id=1")
        if exceeded:
            raise EvalLimit("Provider usage exceeded reservation; campaign stopped")

    def snapshot(self):
        with self.transaction() as db:
            config = dict(db.execute("SELECT * FROM config WHERE id=1").fetchone())
            calls = [dict(row) for row in db.execute("SELECT * FROM calls ORDER BY created,id")]
        return {
            "cap_usd": dollars(config["cap"]),
            "judge_cap_usd": dollars(config["judge_cap"]),
            "estimated_usd": dollars(sum(row["actual"] or 0 for row in calls)),
            "including_uncertain_usd": dollars(sum(row["charged"] for row in calls)),
            "requests": len(calls),
            "stopped": config["stopped"],
            "calls": [
                {
                    **{k: v for k, v in row.items() if k not in {"bound", "charged", "actual", "metadata"}},
                    "bound_usd": dollars(row["bound"]),
                    "charged_usd": dollars(row["charged"]),
                    "actual_usd": dollars(row["actual"]) if row["actual"] is not None else None,
                    "metadata": json.loads(row["metadata"]),
                }
                for row in calls
            ],
        }
