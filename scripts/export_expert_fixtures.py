"""Export synthetic ground truth/context without calling a model."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from expert_eval_cases import CASES, seed_case
from jarvis.config import get_settings
from jarvis.db import engine
from jarvis.models import Base
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


def export(output, repeats=3):
    original_url = get_settings().database_url
    previous = os.environ.get("JARVIS_DATABASE_URL")
    name = "jarvis_expert_eval_" + uuid4().hex
    admin = create_engine(original_url, isolation_level="AUTOCOMMIT")
    created = False
    data = {"kind": "synthetic_ground_truth", "repeats": repeats, "fixtures": []}
    try:
        with admin.connect() as db:
            db.exec_driver_sql(f'CREATE DATABASE "{name}"')
        created = True
        os.environ["JARVIS_DATABASE_URL"] = (
            make_url(original_url).set(database=name).render_as_string(hide_password=False)
        )
        get_settings.cache_clear()
        engine.cache_clear()
        Base.metadata.create_all(engine())
        for repeat in range(1, repeats + 1):
            for case in CASES:
                f = seed_case(case, repeat)
                data["fixtures"].append(f)
        raw = json.dumps(
            data,
            default=lambda x: sorted(x) if isinstance(x, set) else str(x),
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(gzip.compress(raw, mtime=0))
        return {
            "fixtures": len(data["fixtures"]),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    finally:
        engine().dispose()
        engine.cache_clear()
        if created:
            with admin.connect() as db:
                db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()
        if previous is None:
            os.environ.pop("JARVIS_DATABASE_URL", None)
        else:
            os.environ["JARVIS_DATABASE_URL"] = previous
        get_settings.cache_clear()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, choices=range(1, 4), default=3)
    args = p.parse_args()
    print(json.dumps(export(args.output, args.repeats)))
