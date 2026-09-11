from calendar import monthrange
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import select

from .config import get_settings
from .domain import DomainError, advisory, preferences
from .models import BudgetReservation, Usage, now

# Identifies the configured estimation rates, not a provider billing statement.
PRICING_VERSION = "configured-2026-09-11-v2"


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
    uncertain = sum(
        (
            max(Decimal(0), r.amount - (r.actual or Decimal(0)))
            for r in rows
            if r.state == "uncertain" or stale(r)
        ),
        Decimal(0),
    )
    reserved = sum(max(Decimal(0), r.amount - (r.actual or Decimal(0))) for r in rows if r.state != "closed")
    limit = Decimal(str(preferences(db, owner)["monthly_budget_usd"]))
    fraction = float((spent + reserved) / limit) if limit else 1.0
    elapsed_days = max(1, (now() - month).total_seconds() / 86400)
    return {
        "tracking_enabled": True,
        "uncertain_usd": float(uncertain),
        "unconfirmed_sessions": sum(r.state == "uncertain" or stale(r) for r in rows),
        "active_reserved_usd": float(reserved - uncertain),
        "usage_by_model": {model: float(amount) for model, amount in by_model.items()},
        "projected_month_usd": float(spent) / elapsed_days * monthrange(month.year, month.month)[1],
        "budget_mode": "paused"
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
    if available["remaining_usd"] < amount:
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
    if Decimal(str(summary(db, owner)["remaining_usd"])) < increase:
        raise DomainError(
            "BUDGET_EXCEEDED",
            "Model spending is paused at your budget. Saved tasks and reminders still work.",
            402,
        )
    row.amount += increase
    row.last_activity_at = now()
    db.flush()


def record_usage(db, owner, reservation_id, request_id, model, tokens, amount):
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
            tokens={**tokens, "pricing_version": PRICING_VERSION},
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


def stale(row):
    return row.state == "reserved" and (row.last_activity_at or row.created_at) < now() - timedelta(minutes=3)


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


def holds(db, owner):
    if not enabled():
        return []
    rows = db.scalars(
        select(BudgetReservation)
        .where(BudgetReservation.owner_id == owner, BudgetReservation.state != "closed")
        .order_by(BudgetReservation.created_at)
    ).all()
    return [
        {
            "id": r.id,
            "model": r.model,
            "state": "uncertain" if stale(r) else r.state,
            "created_at": r.created_at.isoformat(),
            "last_activity_at": r.last_activity_at.isoformat(),
            "recorded_usd": float(r.actual or 0),
            "held_usd": float(max(Decimal(0), r.amount - (r.actual or Decimal(0)))),
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
    """Explicit operator settlement from provider evidence; never infer zero from age."""
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


def transcription_cost(usage):
    # Duration-based transcription events are distinct from Realtime response tokens.
    seconds = usage.get("seconds")
    if usage.get("type") == "duration" and isinstance(seconds, (int, float)) and seconds >= 0:
        return seconds * 0.017 / 60
    return None
