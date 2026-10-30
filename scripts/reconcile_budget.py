"""Inspect usage buckets and holds, or settle unconfirmed sessions from provider evidence.

No model calls or guessed costs. Without --apply nothing is written. A session without an entry in the
provider evidence stays unconfirmed; it is never cleared as zero.

  one:   --reservation ID --final-usd 0.0123 --evidence "OpenAI usage export 2026-10, req_..." --apply
  many:  --evidence-file statement.json [--apply]
         statement.json: [{"reservation": "ID", "final_usd": "0.0123", "evidence": "..."}, ...]
"""

import argparse
import json
from decimal import Decimal
from pathlib import Path

from jarvis import budget
from jarvis.config import get_settings
from jarvis.db import session_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reservation")
    parser.add_argument("--final-usd", type=Decimal)
    parser.add_argument(
        "--evidence", help="Provider statement/request reference establishing the final amount"
    )
    parser.add_argument("--evidence-file", type=Path, help="Provider evidence for many unconfirmed sessions")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    owner = get_settings().owner_id
    result = {}
    with session_scope() as db:
        if args.evidence_file:
            lookup = budget.evidence_file_lookup(json.loads(args.evidence_file.read_text()))
            result["reconciliation"] = budget.reconcile_unconfirmed(db, owner, lookup, apply=args.apply)
        elif args.apply:
            if not args.reservation or args.final_usd is None or not args.evidence:
                parser.error("--apply requires --reservation, --final-usd and --evidence (or --evidence-file)")
            budget.reconcile(db, owner, args.reservation, args.final_usd, args.evidence)
            db.flush()
        result |= {"budget": budget.summary(db, owner), "holds": budget.holds(db, owner)}
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
