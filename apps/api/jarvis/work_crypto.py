"""Encrypted short-lived execution state, using the existing protected recovery key."""

import json

from cryptography.fernet import Fernet, InvalidToken

from .config import get_settings
from .domain import DomainError


def cipher():
    try:
        return Fernet(get_settings().integration_encryption_key.encode())
    except (ValueError, TypeError):
        raise DomainError(
            "WORK_UNAVAILABLE", "Background work needs its encryption key configured.", 503
        ) from None


def seal(value):
    return cipher().encrypt(json.dumps(value, ensure_ascii=False).encode()).decode()


def unseal(value):
    if not value:
        return {}
    try:
        return json.loads(cipher().decrypt(value.encode()))
    except (InvalidToken, ValueError, TypeError):
        raise DomainError(
            "WORK_UNAVAILABLE", "This saved request could not be opened. Check the recovery key.", 503
        ) from None
