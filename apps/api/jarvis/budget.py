from calendar import monthrange
from decimal import Decimal

from sqlalchemy import select

from .domain import DomainError, advisory, preferences
from .models import BudgetReservation, Usage, now

# Identifies the configured estimation rates, not a provider billing statement.
PRICING_VERSION = "configured-2026-09-11-v1"


def summary(db, owner):
    month = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    rows = db.scalars(
        select(BudgetReservation).where(
            BudgetReservation.owner_id == owner, BudgetReservation.created_at >= month
        )
    ).all()
    usage = list(db.scalars(select(Usage).where(Usage.owner_id == owner, Usage.created_at >= month)))
    spent = sum((u.amount for u in usage), Decimal(0))
    by_model = {}
    for item in usage:
        by_model[item.model] = by_model.get(item.model, Decimal(0)) + item.amount
    uncertain = sum(
        (max(Decimal(0), r.amount - (r.actual or Decimal(0))) for r in rows if r.state == "uncertain"),
        Decimal(0),
    )
    reserved = sum(max(Decimal(0), r.amount - (r.actual or Decimal(0))) for r in rows if r.state != "closed")
    limit = Decimal(str(preferences(db, owner)["monthly_budget_usd"]))
    fraction = float((spent + reserved) / limit) if limit else 1.0
    elapsed_days = max(1, (now() - month).total_seconds() / 86400)
    return {
        "uncertain_usd": float(uncertain),
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
    advisory(db, f"budget:{owner}")
    existing = db.get(BudgetReservation, reservation_id)
    if existing:
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
    db.flush()


def record_usage(db, owner, reservation_id, request_id, model, tokens, amount):
    advisory(db, f"budget:{owner}")
    if db.get(Usage, request_id):
        return
    row = db.get(BudgetReservation, reservation_id)
    cost = Decimal(str(amount))
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


def close(db, owner, reservation_id, uncertain=False):
    advisory(db, f"budget:{owner}")
    row = db.get(BudgetReservation, reservation_id)
    if row:
        # Unknown provider outcomes keep their reservation. Never erase possible spend on a timeout.
        row.state = "uncertain" if uncertain else "closed"


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
