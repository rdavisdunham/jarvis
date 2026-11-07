"""Langfuse tracing of every model call (docs/LANGFUSE.md, "Production tracing").

Off unless both Langfuse keys are set and external services are on. Tracing never changes a model call: every
Langfuse step is guarded, and a failure only logs once.

How traces are organized (one Langfuse project, labeled so everything sorts):
- **Environment:** `JARVIS_DEPLOYMENT_ENVIRONMENT` (local, staging, production).
- **Trace:** one unit of work. A background job (all its model calls), one assistant request, one voice turn.
  `unit()` sets it; calls outside any unit (a search query embedding, say) are a trace of their own.
  Trace name: the job kind or surface (`place_by_meaning`, `record_overview`, `assistant`, `voice`).
- **Observation:** one model call, named by its cost feature (`cost_features.LABELS`, the same lines as the Costs
  page), typed as a generation or an embedding, with the model, input, output, token usage and the recorded cost.
- **Tags:** the cost feature and the surface (`background`, `assistant`, `voice`), so the trace list filters by
  either. **User:** the workspace owner's id. **Session:** the conversation or voice session, when there is one.
"""

import json
import logging
import os
import threading
from contextlib import contextmanager, suppress
from contextvars import ContextVar

from .cost_features import current_feature

logger = logging.getLogger("jarvis.llm_trace")
SURFACES = ("background", "assistant", "voice", "request")
MODEL_KINDS = ("generation", "embedding")

_unit: ContextVar[dict | None] = ContextVar("llm_trace_unit", default=None)
_lock = threading.Lock()
_state: dict = {"client": None, "key": None, "warned": False}


def _warn(message):
    if not _state["warned"]:
        _state["warned"] = True
        logger.warning("Langfuse tracing: %s; model calls continue untraced.", message)


def client():
    """The Langfuse client, or None when tracing is off. Built once per configuration."""
    from .config import get_settings

    settings = get_settings()
    public, secret = settings.langfuse_public_key.strip(), settings.langfuse_secret_key.strip()
    if not (public and secret and settings.langfuse_tracing and settings.external_services_enabled):
        return None
    key = (public, secret, settings.langfuse_base_url, settings.deployment_environment)
    if _state["key"] == key:
        return _state["client"]
    with _lock:
        if _state["key"] != key:
            try:
                from langfuse import Langfuse
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider

                _state["client"] = Langfuse(
                    public_key=public, secret_key=secret, base_url=settings.langfuse_base_url,
                    environment=settings.deployment_environment,
                    release=(os.getenv("RAILWAY_GIT_COMMIT_SHA") or "")[:12] or None,
                    # Its own provider: never the process-wide one, so no other library's spans are exported.
                    tracer_provider=TracerProvider(resource=Resource.create({"service.name": "eridani"})))
            except Exception as error:  # noqa: BLE001 - tracing must never stop the app
                _state["client"] = None
                _warn(f"could not start ({type(error).__name__})")
            _state["key"] = key
    return _state["client"]


def enabled() -> bool:
    return client() is not None


@contextmanager
def unit(name, *, seed, surface="background", owner=None, session=None, metadata=None):
    """Group the model calls made inside into one trace (a job, an assistant request, a voice turn).

    Cheap when nothing calls a model: no trace exists until the first call."""
    token = _unit.set({"name": name, "seed": seed, "surface": surface, "owner": owner, "session": session,
                       "metadata": {k: str(v) for k, v in (metadata or {}).items() if v is not None}})
    try:
        yield
    finally:
        _unit.reset(token)


def current_unit():
    return _unit.get()


# ---------------------------------------------------------------- usage


def _int(value):
    return value if type(value) is int and value >= 0 else None


def usage_details(usage) -> dict:
    """Langfuse usage keys from any of the app's usage shapes (chat, Responses, realtime, embeddings)."""
    if not isinstance(usage, dict):
        return {}
    def pick(*keys):
        return next((_int(usage.get(k)) for k in keys if _int(usage.get(k)) is not None), None)
    out = {"input": pick("prompt_tokens", "input_tokens"), "output": pick("completion_tokens", "output_tokens"),
           "total": pick("total_tokens")}
    inputs = usage.get("prompt_tokens_details") or usage.get("input_token_details") or \
        usage.get("input_tokens_details") or {}
    outputs = usage.get("completion_tokens_details") or usage.get("output_token_details") or \
        usage.get("output_tokens_details") or {}
    if isinstance(inputs, dict):
        out["input_cached_tokens"] = _int(inputs.get("cached_tokens"))
        out["input_cache_write_tokens"] = _int(inputs.get("cache_write_tokens"))
        out["input_audio_tokens"] = _int(inputs.get("audio_tokens"))
    if isinstance(outputs, dict):
        out["output_reasoning_tokens"] = _int(outputs.get("reasoning_tokens"))
        out["output_audio_tokens"] = _int(outputs.get("audio_tokens"))
    seconds = usage.get("seconds")
    if isinstance(seconds, (int, float)) and seconds >= 0:
        out["seconds"] = round(seconds)
    return {k: v for k, v in out.items() if v is not None}


def readable(content):
    """A model's text answer as JSON when it is JSON (structured outputs), else the text itself."""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


# ---------------------------------------------------------------- observations


class _Call:
    """One traced model call. `done()` records the answer; leaving the block without it marks an error."""

    def __init__(self, observation):
        self.observation, self.finished, self._manager = observation, False, None

    def end(self):
        """Close a call opened with `begin()`. A call that never recorded an answer is marked as an error."""
        manager, self._manager = self._manager, None
        if manager is None:
            return
        if not self.finished and self.observation is not None:
            with suppress(Exception):  # Tracing never stops the app.
                self.observation.update(level="ERROR", status_message="No answer was recorded.")
        with suppress(Exception):  # Tracing never stops the app.
            manager.__exit__(None, None, None)

    def done(self, *, output=None, usage=None, cost=None, model=None, metadata=None):
        self.finished = True
        if self.observation is None:
            return
        try:
            fields = {"output": output, "usage_details": usage_details(usage) or None}
            if cost is not None:
                fields["cost_details"] = {"total": float(cost)}
            if model:
                fields["model"] = model
            if metadata:
                fields["metadata"] = metadata
            self.observation.update(**{k: v for k, v in fields.items() if v is not None})
        except Exception as error:  # noqa: BLE001
            _warn(f"could not record a result ({type(error).__name__})")

    def failed(self, error):
        self.finished = True
        if self.observation is None:
            return
        with suppress(Exception):  # Tracing never stops the app.
            self.observation.update(level="ERROR", status_message=f"{type(error).__name__}: {error}"[:500])


def _attributes(feature, surface, owner, session, unit_row):
    from langfuse import propagate_attributes

    tags = sorted({feature, surface} - {None, ""})
    trace_name = (unit_row or {}).get("name") or feature
    metadata = {**((unit_row or {}).get("metadata") or {}), "feature": feature}
    return propagate_attributes(user_id=owner, session_id=session, tags=tags, trace_name=trace_name,
                                metadata={k: str(v)[:200] for k, v in metadata.items()})


@contextmanager
def call(name=None, *, model=None, input=None, kind="generation", feature=None, owner=None, session=None, surface=None,
         model_parameters=None, metadata=None):
    """Trace one model call. `feature` (the cost feature) defaults to the active one, and `name` to the feature;
    the enclosing `unit()` (if any) supplies the trace, owner, session and surface. Always yields a `_Call` (a
    no-op when tracing is off)."""
    lf = client()
    if lf is None:
        yield _Call(None)
        return
    row = _unit.get()
    feature = feature or current_feature.get()
    owner = owner or (row or {}).get("owner")
    session = session or (row or {}).get("session")
    surface = surface or (row or {}).get("surface") or "request"
    handle = None
    try:
        from langfuse import Langfuse

        trace = {"trace_id": Langfuse.create_trace_id(seed=row["seed"])} if row and row.get("seed") else None
        context = _attributes(feature, surface, owner and str(owner), session and str(session), row)
        context.__enter__()
        try:
            fields = {"model": model, "model_parameters": model_parameters} if kind in MODEL_KINDS else {}
            handle = lf.start_as_current_observation(
                name=name or feature, as_type=kind, trace_context=trace, input=input, metadata=metadata,
                **{k: v for k, v in fields.items() if v is not None})
            observation = handle.__enter__()
        except Exception:
            context.__exit__(None, None, None)
            raise
    except Exception as error:  # noqa: BLE001
        _warn(f"could not open a trace ({type(error).__name__})")
        yield _Call(None)
        return
    traced = _Call(observation)
    try:
        yield traced
    except BaseException as error:
        traced.failed(error)
        raise
    finally:
        with suppress(Exception):  # Tracing never stops the app.
            handle.__exit__(None, None, None)
        with suppress(Exception):  # Tracing never stops the app.
            context.__exit__(None, None, None)


def step(name, *, input=None, kind="tool", metadata=None):
    """Trace a non-model step inside a unit, such as one assistant tool call, timed and next to the model calls
    that asked for it. Yields a `_Call`: `done(output=...)` records the result."""
    return call(name, kind=kind, input=input, metadata=metadata)


def begin(name=None, **kwargs):
    """`call()` for code that can't nest a with-block around the call: always `.end()` it in a `finally`."""
    manager = call(name, **kwargs)
    traced = manager.__enter__()
    traced._manager = manager
    return traced


def record(name=None, *, model, input=None, output=None, usage=None, cost=None, kind="generation", feature=None,
           owner=None, session=None, surface=None, metadata=None):
    """Trace a model call that already happened (a realtime voice response reported by an event)."""
    with call(name, model=model, input=input, kind=kind, feature=feature, owner=owner, session=session,
              surface=surface, metadata=metadata) as traced:
        traced.done(output=output, usage=usage, cost=cost)


def flush():
    lf = _state["client"]
    if lf is not None:
        with suppress(Exception):  # Tracing never stops the app.
            lf.flush()
