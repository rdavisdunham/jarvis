"""Export the fixed held-out synthetic questions/state without provider calls."""

import argparse
import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import reliability_eval_cases as cases
from evaluate_expert_agents import DATABASE_PREFIX, isolated_environment, validate_disposable_database
from jarvis.config import get_settings
from jarvis.db import engine
from jarvis.models import Base
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


def json_value(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=json_value).encode()
    ).hexdigest()


def export_fixtures(output, repeats=3):
    original = get_settings().database_url
    name = DATABASE_PREFIX + uuid4().hex
    database_url = make_url(original).set(database=name).render_as_string(hide_password=False)
    admin = create_engine(original, isolation_level="AUTOCOMMIT")
    created = False
    report = {
        "suite": "reliability8",
        "generated_at": datetime.now(UTC).isoformat(),
        "paid_provider_calls": 0,
        "repeats": repeats,
        "fixtures": [],
        "database_removed": False,
        "deterministic_reseed_verified": False,
    }
    try:
        with admin.connect() as db:
            validate_disposable_database(name, name)
            db.exec_driver_sql(f'CREATE DATABASE "{name}"')
        created = True
        with isolated_environment(database_url):
            Base.metadata.create_all(engine())
            for repeat in range(1, repeats + 1):
                for case in cases.CASES:
                    fixture = cases.seed_case(case, repeat)
                    fixture.pop("tools")
                    fixture.pop("integration_calls")
                    fixture.pop("ui_trace")
                    payload = json.loads(json.dumps(fixture, ensure_ascii=False, default=json_value))
                    again = cases.seed_case(case, repeat)
                    for field in ("tools", "integration_calls", "ui_trace"):
                        again.pop(field)
                    assert fingerprint(payload) == fingerprint(again), (case, repeat)
                    report["fixtures"].append(
                        {
                            "case": case,
                            "repeat": repeat,
                            "seed_payload_sha256": fingerprint(payload),
                            "fixture": payload,
                        }
                    )
            report["deterministic_reseed_verified"] = True
    finally:
        if created:
            validate_disposable_database(name, name)
            with admin.connect() as db:
                db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
            report["database_removed"] = True
        admin.dispose()
    sources = (
        "scripts/export_reliability_fixtures.py",
        "scripts/reliability_eval_cases.py",
        "scripts/eval_integrations.py",
        "scripts/eval_ui.py",
        "docs/RELIABILITY_HELDOUT_EVAL_PLAN.md",
    )
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new fixture artifact path; existing exports are immutable.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        gzip.compress(json.dumps(report, ensure_ascii=False, default=json_value).encode(), mtime=0)
    )
    print(
        json.dumps(
            {
                "path": str(output),
                "fixtures": len(report["fixtures"]),
                "deterministic_reseed_verified": report["deterministic_reseed_verified"],
                "database_removed": report["database_removed"],
                "paid_provider_calls": 0,
            }
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".runtime/reliability-heldout-fixtures.json.gz"))
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    export_fixtures(args.output, args.repeats)
