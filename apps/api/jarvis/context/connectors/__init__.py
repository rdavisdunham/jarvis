"""Connector registry: a connector key resolves to an adapter; every adapter ends in `sources.receive`.

`upload` and `api` are pass-throughs. `granola` also has an optional pull poller; `zoom` is a documented stub.
Any other key (`recorder`, `other`, a future tool) resolves to `api` handling, and the original label is kept
in `envelope.metadata["connector_label"]`, so the pipeline never needs to know which client delivered a source.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..schema import CONNECTORS, SourceEnvelope

LABEL_KEY = "connector_label"


@dataclass(frozen=True)
class Adapter:
    key: str  # the envelope connector value (one of schema.CONNECTORS)
    description: str
    poll: Callable[..., Any] | None = None  # sync(db, owner, connection) for pull adapters
    backfill: Callable[..., Any] | None = None


def _granola_sync(*args, **kwargs):
    from . import granola
    return granola.sync(*args, **kwargs)


def _granola_backfill(*args, **kwargs):
    from . import granola
    return granola.backfill(*args, **kwargs)


def _zoom_sync(*args, **kwargs):
    from . import zoom
    return zoom.sync(*args, **kwargs)


ADAPTERS: dict[str, Adapter] = {
    "upload": Adapter("upload", "Files uploaded on the Sources settings page"),
    "api": Adapter("api", "Any external client using PUT /external/sources/{connector}/{external_id}"),
    "granola": Adapter("granola", "Granola REST poller (Business workspaces; off unless enabled)",
                       poll=_granola_sync, backfill=_granola_backfill),
    "zoom": Adapter("zoom", "Zoom cloud-recording transcripts (not implemented; submit exports through api)",
                    poll=_zoom_sync),
}
assert set(ADAPTERS) == set(CONNECTORS)


def normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def resolve(label: str) -> Adapter:
    """The adapter for a connector key; unknown keys get the generic `api` handling."""
    return ADAPTERS.get(normalize_label(label), ADAPTERS["api"])


def prepare(label: str, envelope: SourceEnvelope) -> SourceEnvelope:
    """The envelope as the registry stores it: connector set to the resolved adapter key, and the client's
    label kept in metadata when it differs."""
    name = normalize_label(label)
    adapter = resolve(name)
    metadata = dict(envelope.metadata)
    if name and name != adapter.key:
        metadata[LABEL_KEY] = name[:50]
    if envelope.connector == adapter.key and metadata == envelope.metadata:
        return envelope
    return SourceEnvelope.model_validate({**envelope.model_dump(), "connector": adapter.key, "metadata": metadata})


def receive(db, owner: str, label: str, envelope: SourceEnvelope, connection_id: str | None = None):
    """Push delivery through the registry: resolve the label, then hand the envelope to `sources.receive`."""
    from .. import sources
    return sources.receive(db, owner, prepare(label, envelope), connection_id=connection_id)
