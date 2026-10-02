"""External network prohibition for deterministic in-process adapters."""

from contextlib import contextmanager
from unittest.mock import patch

import httpx

from evals import offline as sockets


@contextmanager
def offline():
    def deny(*args, **kwargs):
        raise RuntimeError("Offline adapter attempted an external service call")

    sockets.pytest_configure(None)
    try:
        with patch.object(httpx.Client, "send", deny), patch.object(httpx.AsyncClient, "send", deny):
            yield
    finally:
        sockets.pytest_unconfigure(None)
