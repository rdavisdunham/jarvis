"""Connector registry: every pilot key resolves, unknown labels fall back to `api` with the label kept."""

import sys
import types

import jarvis.context as context_pkg
import pytest
from jarvis.context import connectors
from jarvis.context.connectors import granola, zoom
from jarvis.context.schema import CONNECTORS


@pytest.fixture
def received(monkeypatch):
    calls = []
    module = getattr(context_pkg, "sources", None) or sys.modules.get("jarvis.context.sources")
    if module is None:
        module = types.ModuleType("jarvis.context.sources")
        monkeypatch.setitem(sys.modules, "jarvis.context.sources", module)
        monkeypatch.setattr(context_pkg, "sources", module, raising=False)
    monkeypatch.setattr(module, "receive", lambda db, owner, envelope, connection_id=None:
                        calls.append((db, owner, envelope, connection_id)) or "result", raising=False)
    return calls


@pytest.mark.parametrize("key", CONNECTORS)
def test_every_pilot_key_resolves_to_itself(key):
    assert connectors.resolve(key).key == key
    assert connectors.resolve(key.upper() + " ").key == key


@pytest.mark.parametrize("label", ["recorder", "other", "fireflies", ""])
def test_unknown_labels_resolve_to_api(label):
    assert connectors.resolve(label) is connectors.ADAPTERS["api"]


def test_prepare_keeps_the_label(envelope):
    prepared = connectors.prepare("Recorder", envelope)
    assert prepared.connector == "api" and prepared.metadata[connectors.LABEL_KEY] == "recorder"
    assert prepared.segments == envelope.segments
    assert connectors.prepare("api", envelope) is envelope
    zoomed = connectors.prepare("zoom", envelope)
    assert zoomed.connector == "zoom" and connectors.LABEL_KEY not in zoomed.metadata


def test_receive_passes_through_to_sources(received, envelope):
    assert connectors.receive("db", "davin", "upload", envelope.model_copy(update={"connector": "upload"}),
                              connection_id="conn-1") == "result"
    assert connectors.receive("db", "davin", "recorder", envelope) == "result"
    (_, owner, first, connection_id), (_, _, second, _) = received
    assert (owner, first.connector, connection_id) == ("davin", "upload", "conn-1")
    assert second.connector == "api" and second.metadata["connector_label"] == "recorder"


def test_pull_adapters(monkeypatch):
    monkeypatch.setattr(granola, "sync", lambda *a, **k: ("synced", a))
    monkeypatch.setattr(granola, "backfill", lambda *a, **k: ("backfilled", a))
    adapter = connectors.resolve("granola")
    assert adapter.poll("db", "davin", "conn") == ("synced", ("db", "davin", "conn"))
    assert adapter.backfill("db", "davin", "conn") == ("backfilled", ("db", "davin", "conn"))
    assert connectors.resolve("api").poll is None and connectors.resolve("upload").poll is None
    with pytest.raises(NotImplementedError):
        connectors.resolve("zoom").poll("db", "davin", "conn")
    with pytest.raises(NotImplementedError):
        zoom.to_envelope({}, "WEBVTT")
