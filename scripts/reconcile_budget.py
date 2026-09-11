"""Inspect holds or settle one from provider evidence. No model calls or guessed costs."""

import argparse
import json
from decimal import Decimal

from jarvis import budget
from jarvis.config import get_settings
from jarvis.db import session_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation")
    parser.add_argument("--final-usd", type=Decimal)
    parser.add_argument(
        "--evidence", help="Provider statement/request reference establishing the final amount"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    owner = get_settings().owner_id
    with session_scope() as db:
        if args.apply:
            if not args.reservation or args.final_usd is None or not args.evidence:
                parser.error("--apply requires --reservation, --final-usd and --evidence")
            budget.reconcile(db, owner, args.reservation, args.final_usd, args.evidence)
            db.flush()
        print(json.dumps({"budget": budget.summary(db, owner), "holds": budget.holds(db, owner)}, indent=2))


if __name__ == "__main__":
    main()
