"""Actionable relationship failures without revealing another owner's records."""

from .domain import DomainError, owned


def reference(db, owner, model, value, field):
    try:
        return owned(db, model, value, owner)
    except DomainError as exc:
        if exc.code not in {"NOT_FOUND", "MALFORMED_ID"}:
            raise
        kind = model.__name__.lower()
        raise DomainError(
            "INVALID_REFERENCE",
            f"The {kind} reference in {field} is unavailable. Look up that {kind} again; "
            "send only the fields being changed. The rejected change saved nothing.",
            404,
            {"field": field, "reference_type": kind, "invalid_id": value,
             "recovery": "Read the current target, correct this field, and retry with the current revision. "
                         "Omit unchanged relationship fields; do not erase links to work around an error."},
        ) from None
