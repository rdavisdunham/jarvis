from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import get_settings


@lru_cache
def engine():
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=15,
        connect_args={"connect_timeout": 10},
    )


def session_factory():
    return sessionmaker(bind=engine(), expire_on_commit=False)


def session_scope():
    return session_factory().begin()
