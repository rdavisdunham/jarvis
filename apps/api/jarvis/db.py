from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from .config import get_settings


@lru_cache
def engine():
    settings = get_settings()
    result = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=15,
        # Receipts, revert checks and API timestamps assume UTC session rendering.
        connect_args={"connect_timeout": 10, "options": "-c timezone=UTC"},
    )

    @event.listens_for(result, "connect")
    def utc(connection, _record):
        # libpq's PGTZ overrides startup options; pin again once connected.
        autocommit = connection.autocommit
        connection.autocommit = True
        connection.execute("SET TIME ZONE 'UTC'")
        connection.autocommit = autocommit

    return result


def session_factory():
    return sessionmaker(bind=engine(), expire_on_commit=False)


def session_scope():
    return session_factory().begin()
