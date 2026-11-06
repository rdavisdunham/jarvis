"""Model usage in three separate buckets.

recorded     Usage rows: provider-reported (or provider-evidence-reconciled) charges.
reserved     Open reservations for work that is still active.
unconfirmed  Reservations whose provider outcome is unknown (timeout, lost connection, missing final
             usage). Their held allowance counts against the limit until provider evidence settles them
             or they lapse.

Lapsed holds stop counting and the worker sweep closes them, keeping the rows for audit: holds created
before cost_tracking_since ("predates_tracking"), and unconfirmed holds older than UNCONFIRMED_DAYS
("expired_without_evidence", nothing held). By owner decision a month-old hold with no provider record
is not worth carrying; provider evidence can still reconcile a lapsed row explicitly.
"""

from calendar import monthrange
from datetime import UTC, timedelta
from decimal import Decimal, InvalidOperation

from sqlalchemy import select

from .config import get_settings
from .cost_features import LABELS, current_feature
from .domain import DomainError, advisory, preferences
from .models import BudgetReservation, Usage, now

# Identifies the configured estimation rates, not a provider billing statement.
PRICING_VERSION = "configured-2026-10-04-v4"
UNCONFIRMED_DAYS = 30


def enabled():
    return get_settings().cost_tracking_enabled


def summary(db, owner):
    if not enabled():
        return {
            "tracking_enabled": False,
            "budget_mode": "disabled",
            "spent_usd": None,
            "uncertain_usd": None,
            "active_reserved_usd": None,
            "projected_month_usd": None,
            "reserved_usd": None,
            "remaining_usd": None,
            "limit_usd": None,
            "usage_by_model": {},
            "unconfirmed_sessions": None,
            "buckets": None,
            "pricing_version": None,
            "month": now().strftime("%Y-%m"),
            "approximate": False,
        }
    month = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    rows = db.scalars(
        select(BudgetReservation).where(
            BudgetReservation.owner_id == owner,
            (BudgetReservation.created_at >= month) | (BudgetReservation.state != "closed"),
        )
    ).all()
    usage = list(db.scalars(select(Usage).where(Usage.owner_id == owner, Usage.created_at >= month)))
    spent = sum((u.amount for u in usage), Decimal(0))
    by_model = {}
    for item in usage:
        by_model[item.model] = by_model.get(item.model, Decimal(0)) + item.amount
    rows = [r for r in rows if r.state == "closed" or not lapse_reason(r)]
    unconfirmed = [r for r in rows if bucket(r) == "unconfirmed"]
    historic = [r for r in unconfirmed if r.created_at < month]
    uncertain = sum((held(r) for r in unconfirmed), Decimal(0))
    reserved = sum((held(r) for r in rows if r.state != "closed"), Decimal(0))
    limit = Decimal(str(preferences(db, owner)["monthly_budget_usd"]))
    fraction = float((spent + reserved) / limit) if limit else 1.0
    elapsed_days = max(1, (now() - month).total_seconds() / 86400)
    return {
        "tracking_enabled": True,
        "enforcement_enabled": get_settings().budget_enforcement_enabled,
        "report": usage_report(db, owner),
        "uncertain_usd": float(uncertain),
        "unconfirmed_sessions": len(unconfirmed),
        "buckets": {
            "recorded": {"usd": float(spent), "records": len(usage)},
            "reserved": {
                "usd": float(reserved - uncertain),
                "sessions": sum(bucket(r) == "reserved" for r in rows),
            },
            "unconfirmed": {
                "usd": float(uncertain),
                "sessions": len(unconfirmed),
                "historic_usd": float(sum((held(r) for r in historic), Decimal(0))),
                "historic_sessions": len(historic),
                "oldest_at": min(r.created_at for r in unconfirmed).isoformat() if unconfirmed else None,
            },
        },
        "active_reserved_usd": float(reserved - uncertain),
        "usage_by_model": {model: float(amount) for model, amount in by_model.items()},
        "projected_month_usd": (
            None if get_settings().cost_tracking_since and get_settings().cost_tracking_since.astimezone(UTC) > month
            else float(spent) / elapsed_days * monthrange(month.year, month.month)[1]
        ),
        "budget_mode": "tracking_only"
        if not get_settings().budget_enforcement_enabled
        else "paused"
        if fraction >= 1
        else "defer_optional"
        if fraction >= 0.95
        else "warning"
        if fraction >= 0.8
        else "normal",
        "pricing_version": PRICING_VERSION,
        "spent_usd": float(spent),
        "reserved_usd": float(reserved),
        "limit_usd": float(limit),
        "remaining_usd": float(max(Decimal(0), limit - spent - reserved)),
        "month": month.strftime("%Y-%m"),
        "approximate": True,
    }


def usage_report(db, owner):
    """Rolling recorded usage. Missing history is never extrapolated as zero."""
    end = now()
    week, month = end - timedelta(days=7), end - timedelta(days=30)
    rows = db.scalars(select(Usage).where(
        Usage.owner_id == owner, Usage.created_at >= month, Usage.created_at <= end
    )).all()
    features = {}
    for row in rows:
        key = (row.tokens or {}).get("feature", "unattributed")
        key = key if key in LABELS else "unattributed"
        item = features.setdefault(key, {
            "id": key, "label": LABELS[key], "last_7_days_usd": Decimal(0),
            "last_30_days_usd": Decimal(0), "usage_records": 0,
        })
        item["last_30_days_usd"] += row.amount
        if row.created_at >= week:
            item["last_7_days_usd"] += row.amount
        item["usage_records"] += 1
    values = sorted(features.values(), key=lambda f: (-f["last_30_days_usd"], f["id"]))
    since = get_settings().cost_tracking_since
    return {
        "as_of": end.isoformat(),
        "tracking_since": since.isoformat() if since else None,
        "last_7_days_usd": float(sum((f["last_7_days_usd"] for f in values), Decimal(0))),
        "last_30_days_usd": float(sum((f["last_30_days_usd"] for f in values), Decimal(0))),
        "partial_7_days": since is None or since.astimezone(UTC) > week,
        "partial_30_days": since is None or since.astimezone(UTC) > month,
        "features": [{**f, "last_7_days_usd": float(f["last_7_days_usd"]),
                      "last_30_days_usd": float(f["last_30_days_usd"])} for f in values],
    }


def reserve(db, owner, reservation_id, amount, model, *, optional=False):
    if not enabled():
        return
    advisory(db, f"budget:{owner}")
    existing = db.get(BudgetReservation, reservation_id)
    if existing:
        if existing.owner_id != owner:
            raise DomainError("NOT_FOUND", "Reservation not found.", 404)
        return existing
    available = summary(db, owner)
    if optional and available["budget_mode"] in {"defer_optional", "paused"}:
        raise DomainError(
            "BUDGET_DEFERRED", "Automatic memory work is paused near the monthly model budget.", 402
        )
    if get_settings().budget_enforcement_enabled and available["remaining_usd"] < amount:
        raise DomainError(
            "BUDGET_EXCEEDED", "The model budget is reserved or used. Tasks and reminders still work.", 402
        )
    row = BudgetReservation(
        id=reservation_id, owner_id=owner, amount=Decimal(str(amount)), actual=Decimal(0), model=model
    )
    db.add(row)
    db.flush()
    return row


def ensure_room(db, owner, reservation_id, headroom):
    """Reserve a conservative bound before EACH provider request, including tool continuations."""
    if not enabled():
        return
    advisory(db, f"budget:{owner}")
    row = db.get(BudgetReservation, reservation_id)
    if not row or row.owner_id != owner or row.state == "closed":
        raise DomainError("BUDGET_EXCEEDED", "This model session has ended.", 402)
    target = (row.actual or Decimal(0)) + Decimal(str(headroom))
    increase = max(Decimal(0), target - row.amount)
    if get_settings().budget_enforcement_enabled and Decimal(str(summary(db, owner)["remaining_usd"])) < increase:
        raise DomainError(
            "BUDGET_EXCEEDED",
            "Model spending is paused at your budget. Saved tasks and reminders still work.",
            402,
        )
    row.amount += increase
    row.last_activity_at = now()
    db.flush()


def record_usage(db, owner, reservation_id, request_id, model, tokens, amount, *, feature=None):
    if not enabled():
        return
    advisory(db, f"budget:{owner}")
    if db.get(Usage, request_id):
        return
    row = db.get(BudgetReservation, reservation_id)
    if not row or row.owner_id != owner:
        raise DomainError("NOT_FOUND", "Reservation not found.", 404)
    cost = Decimal(str(amount))
    if not cost.is_finite() or cost < 0:
        raise DomainError("INVALID_ARGUMENT", "Usage must be a finite nonnegative amount.")
    row.last_activity_at = now()
    db.add(
        Usage(
            request_id=request_id,
            owner_id=owner,
            reservation_id=reservation_id,
            model=model,
            tokens={**tokens, "pricing_version": PRICING_VERSION, "feature": feature or current_feature.get()},
            amount=cost,
        )
    )
    row.actual = (row.actual or Decimal(0)) + cost


def close(db, owner, reservation_id, uncertain=False, reason=None):
    if not enabled():
        return
    advisory(db, f"budget:{owner}")
    row = db.get(BudgetReservation, reservation_id)
    if row and row.owner_id == owner:
        # Unknown provider outcomes keep their reservation. Never erase possible spend on a timeout.
        row.state = "uncertain" if uncertain else "closed"
        row.closed_at = now()
        row.settlement = {
            "reason": reason or ("provider_outcome_unknown" if uncertain else "provider_work_accounted"),
            "at": now().isoformat(),
        }


def realtime_cost(usage, model):
    inputs = usage.get("input_token_details", {})
    outputs = usage.get("output_token_details", {})
    cached = inputs.get("cached_tokens_details", {})
    mini = "mini" in model
    audio_in, audio_out = (10, 20) if mini else (32, 64)
    text_in, text_out = (0.6, 2.4) if mini else (4, 24)
    cached_audio, cached_text = cached.get("audio_tokens", 0), cached.get("text_tokens", 0)
    return (
        max(0, inputs.get("audio_tokens", 0) - cached_audio) * audio_in
        + max(0, inputs.get("text_tokens", 0) - cached_text) * text_in
        + outputs.get("audio_tokens", 0) * audio_out
        + outputs.get("text_tokens", 0) * text_out
        + cached_audio * (0.3 if mini else 0.4)
        + cached_text * (0.06 if mini else 0.4)
    ) / 1_000_000


# What one more Realtime response can add, on top of the context it reads: a long spoken turn (two minutes
# at ~600 audio tokens a minute), a large tool result, the reply cap, and the turn's transcription.
REALTIME_TURN = {"audio": 1_200, "text": 4_000}
REALTIME_REPLY_TOKENS = 1_024  # session max_output_tokens
REALTIME_TURN_TRANSCRIPTION_USD = 120 * 0.017 / 60
# Before any response reports usage: instructions, tools, memory context and eight restored messages.
REALTIME_OPENING_CONTEXT = {"audio": 0, "text": 32_000}


def realtime_headroom(model, context=None):
    """A ceiling for one more Realtime response, priced as if nothing were cached.

    `context` is the largest {"audio", "text"} token context a response in this session has read so far
    (input plus its own output, which the next response reads back). Replaces a flat 128k-token bound,
    so a lost session holds tens of cents instead of dollars.
    """
    context = context or REALTIME_OPENING_CONTEXT
    usage = {
        "input_token_details": {
            "audio_tokens": context["audio"] + REALTIME_TURN["audio"],
            "text_tokens": context["text"] + REALTIME_TURN["text"],
        },
        "output_token_details": {"audio_tokens": REALTIME_REPLY_TOKENS},
    }
    return realtime_cost(usage, model) + REALTIME_TURN_TRANSCRIPTION_USD


def realtime_context(usage):
    """The token context a response read plus what it added, from its reported usage."""
    inputs, outputs = usage.get("input_token_details", {}), usage.get("output_token_details", {})
    return {
        "audio": inputs.get("audio_tokens", 0) + outputs.get("audio_tokens", 0),
        "text": inputs.get("text_tokens", 0) + outputs.get("text_tokens", 0),
    }


def stale(row):
    return row.state == "reserved" and (row.last_activity_at or row.created_at) < now() - timedelta(minutes=3)


def bucket(row):
    """reserved (active work), unconfirmed (outcome unknown; a stale lease counts) or settled."""
    if row.state == "closed":
        return "settled"
    return "unconfirmed" if row.state == "uncertain" or stale(row) else "reserved"


def held(row):
    return max(Decimal(0), row.amount - (row.actual or Decimal(0)))


def touch(db, owner, reservation_id):
    if not enabled():
        return
    advisory(db, f"budget:{owner}")
    row = db.get(BudgetReservation, reservation_id)
    if row and row.owner_id == owner and row.state == "reserved":
        row.last_activity_at = now()


def expire_abandoned(db, owner):
    if not enabled():
        return 0
    advisory(db, f"budget:{owner}")
    rows = db.scalars(
        select(BudgetReservation).where(
            BudgetReservation.owner_id == owner,
            BudgetReservation.state == "reserved",
            BudgetReservation.last_activity_at < now() - timedelta(minutes=3),
        )
    ).all()
    for row in rows:
        close(db, owner, row.id, uncertain=True, reason="activity_lease_expired")
    return len(rows)


def lapse_reason(row):
    """Why an open hold no longer counts, or None. sweep_lapsed closes it with this reason."""
    if row.state == "closed":
        return None
    since = get_settings().cost_tracking_since
    if since and row.created_at < since:
        return "predates_tracking"
    if bucket(row) == "unconfirmed" and row.created_at < now() - timedelta(days=UNCONFIRMED_DAYS):
        return "expired_without_evidence"
    return None


def sweep_lapsed(db, owner):
    """Close holds that no longer count. The rows stay; the settlement notes what was still held."""
    if not enabled():
        return 0
    advisory(db, f"budget:{owner}")
    rows = db.scalars(
        select(BudgetReservation).where(
            BudgetReservation.owner_id == owner, BudgetReservation.state != "closed"
        )
    ).all()
    closed = 0
    for row in rows:
        reason = lapse_reason(row)
        if not reason:
            continue
        row.settlement = {
            "reason": reason,
            "held_usd": str(held(row)),
            "previous_state": row.state,
            "at": now().isoformat(),
        }
        row.state, row.closed_at = "closed", now()
        closed += 1
    db.flush()
    return closed


def holds(db, owner):
    if not enabled():
        return []
    rows = db.scalars(
        select(BudgetReservation)
        .where(BudgetReservation.owner_id == owner, BudgetReservation.state != "closed")
        .order_by(BudgetReservation.created_at)
    ).all()
    rows = [r for r in rows if not lapse_reason(r)]
    month = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return [
        {
            "id": r.id,
            "model": r.model,
            "state": "uncertain" if stale(r) else r.state,
            "bucket": bucket(r),
            "historic": r.created_at < month,
            "created_at": r.created_at.isoformat(),
            "last_activity_at": r.last_activity_at.isoformat(),
            "recorded_usd": float(r.actual or 0),
            "held_usd": float(held(r)),
            "reason": r.settlement.get(
                "reason",
                "older_session_without_final_evidence"
                if r.state == "uncertain" or stale(r)
                else "active_work",
            ),
        }
        for r in rows
    ]


def reconcile(db, owner, reservation_id, final_amount, evidence):
    """Explicit operator settlement from provider evidence. Also works on a lapsed (swept) hold."""
    if not enabled():
        raise DomainError("TRACKING_DISABLED", "Cost tracking is disabled during development.")
    advisory(db, f"budget:{owner}")
    row = db.get(BudgetReservation, reservation_id)
    if not row or row.owner_id != owner:
        raise DomainError("NOT_FOUND", "Reservation not found.", 404)
    amount = Decimal(str(final_amount))
    if not amount.is_finite() or amount < 0 or not evidence.strip():
        raise DomainError(
            "INVALID_ARGUMENT", "A final nonnegative amount and provider evidence are required."
        )
    if row.state == "reserved" and not stale(row):
        raise DomainError("REVISION_CONFLICT", "This request is still active.", 409)
    if row.settlement.get("reason") == "provider_evidence":
        if row.settlement["final_usd"] != str(amount) or row.settlement["evidence"] != evidence.strip():
            raise DomainError(
                "REVISION_CONFLICT", "This reservation was already reconciled differently.", 409
            )
        return
    # Retain every original usage event. The signed adjustment reconciles the total.
    adjustment = amount - (row.actual or Decimal(0))
    db.add(
        Usage(
            request_id=f"reconcile:{row.id}",
            owner_id=owner,
            reservation_id=row.id,
            model=row.model,
            tokens={"basis": "provider_reconciliation", "evidence": evidence.strip()},
            amount=adjustment,
            created_at=row.created_at,
        )
    )
    row.actual = amount
    row.state = "closed"
    row.closed_at = now()
    row.settlement = {
        "reason": "provider_evidence",
        "evidence": evidence.strip(),
        "final_usd": str(amount),
        "at": now().isoformat(),
    }


def reconcile_unconfirmed(db, owner, lookup, *, apply=True):
    """Settle unconfirmed sessions one by one, only from provider evidence.

    `lookup(hold)` gets a holds() entry and returns {"final_usd", "evidence"} from a provider statement,
    or None when the provider has nothing for it. Without evidence a session stays unconfirmed until it
    lapses (see sweep_lapsed); recorded charges and active reservations are not touched. A settlement's
    adjustment Usage row is dated at the reservation, so a historic session lands in its own month.
    """
    if not enabled():
        raise DomainError("TRACKING_DISABLED", "Cost tracking is disabled during development.")
    report = {"settled": [], "unconfirmed": [], "rejected": []}
    for hold in holds(db, owner):
        if hold["bucket"] != "unconfirmed":
            continue
        try:
            found = lookup(hold)
        except Exception as exc:  # noqa: BLE001 - a failed lookup is not evidence
            report["unconfirmed"].append({**hold, "why": "lookup_failed:" + type(exc).__name__})
            continue
        if not found:
            report["unconfirmed"].append({**hold, "why": "no_provider_evidence"})
            continue
        evidence = str(found.get("evidence") or "").strip()
        try:
            final = Decimal(str(found.get("final_usd")))
        except (InvalidOperation, ValueError):
            final = None
        if final is None or not final.is_finite() or final < 0 or not evidence:
            report["rejected"].append({**hold, "why": "incomplete_evidence"})
            continue
        if apply:
            try:
                reconcile(db, owner, hold["id"], final, evidence)
            except DomainError as exc:
                report["rejected"].append({**hold, "why": exc.code})
                continue
        report["settled"].append({**hold, "final_usd": str(final), "evidence": evidence})
    if apply:
        db.flush()
    return report


def evidence_file_lookup(entries):
    """A lookup over an operator's provider statement: [{"reservation", "final_usd", "evidence"}, ...].
    Every entry names one reservation; nothing is matched by model, date or amount."""
    by_id = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("reservation"):
            raise ValueError("Each evidence entry needs a reservation id.")
        if entry["reservation"] in by_id:
            raise ValueError(f"Duplicate evidence for reservation {entry['reservation']}.")
        by_id[entry["reservation"]] = entry
    return lambda hold: by_id.get(hold["id"])


def transcription_cost(usage):
    # Duration-based transcription events are distinct from Realtime response tokens.
    seconds = usage.get("seconds")
    if usage.get("type") == "duration" and isinstance(seconds, (int, float)) and seconds >= 0:
        return seconds * 0.017 / 60
    return None
