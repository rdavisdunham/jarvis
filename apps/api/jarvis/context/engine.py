"""The four replaceable engine operations (plan section 2). Implementations: the native engine and
`context.doubles`. There is no registry or switching; callers construct the engine they use.

Engines receive authorized context through the application boundary and return proposals only: they never
write canonical tables, create tasks, notify or grant access. Core validates every result before applying it.
"""

from typing import Protocol, runtime_checkable

from .schema import (
    BriefSection,
    ExtractionResult,
    ExtractRequest,
    Hit,
    ReconcileOp,
    ReconcileRequest,
    RetrieveQuery,
    SynthesizeRequest,
)


@runtime_checkable
class Extractor(Protocol):
    def extract(self, request: ExtractRequest) -> ExtractionResult: ...


@runtime_checkable
class Reconciler(Protocol):
    def reconcile(self, request: ReconcileRequest) -> list[ReconcileOp]: ...


@runtime_checkable
class Retriever(Protocol):
    """Applies `query.scope` inside candidate selection, before ranking, so top-k recall is not lost."""

    def retrieve(self, query: RetrieveQuery) -> list[Hit]: ...


@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(self, request: SynthesizeRequest) -> list[BriefSection]: ...
