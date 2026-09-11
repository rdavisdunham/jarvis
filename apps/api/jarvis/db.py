from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import get_settings


@lru_cache
def engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=10, max_overflow=10)


def session_factory():
    return sessionmaker(bind=engine(), expire_on_commit=False)


def session_scope():
    return session_factory().begin()
