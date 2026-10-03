from sqlalchemy import text

CHANNEL = "eridani_outbox"


def wake_dispatch(db):
    """A commit-delivered hint only. The transactional outbox remains authoritative."""
    db.execute(text("SELECT pg_notify(:channel, '')"), {"channel": CHANNEL})
