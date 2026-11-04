"""Meeting-context REST twins of the MCP tools (`context/mcp.py`): /api/v1/external/context/*.

Scopes: `context:read` for every read, `context:write` for feedback (a suggestion only). Each route refuses when
`context_enabled` is off (CONTEXT_DISABLED) and in shared workspaces (PERSONAL_WORKSPACE); `service` authorizes
before assembly and again before returning. Sync handlers run in the threadpool, where the service embeds a
search query with no transaction open.
"""

from typing import Annotated

from fastapi import APIRouter, Path, Query

from .. import service
from .sources import Bot

router = APIRouter(prefix="/context", tags=["Meeting context"])


@router.post("/search")
def search(body: service.SearchInput, identity: Bot):
    """`context_search`: ranked, cited context for a question and/or an entity."""
    return service.search(body)


@router.post("/brief")
def brief(body: service.BriefInput, identity: Bot):
    """`context_brief`: an entity's evidence-cited brief, or a brief about a question."""
    return service.brief(body)


@router.get("/evidence/{assertion_id}")
def evidence(assertion_id: Annotated[str, Path(max_length=100)], identity: Bot, include_history: bool = False):
    """`context_evidence`: every supporting and contradicting span for one assertion."""
    return service.evidence_for(service.EvidenceInput(assertion_id=assertion_id, include_history=include_history))


@router.get("/sources/{source_id}")
def source(source_id: Annotated[str, Path(max_length=100)], identity: Bot, segment_start: int = Query(0, ge=0),
           segment_end: int | None = Query(None, ge=0), revision: int | None = Query(None, ge=1)):
    """`context_source`: a segment range of an authorized source revision."""
    return service.source(service.SourceInput(source_id=source_id, segment_start=segment_start,
                                              segment_end=segment_end, revision=revision))


@router.post("/open-loops")
def open_loops(body: service.OpenLoopsInput, identity: Bot):
    """`context_open_loops`: open commitments, the user's own or waiting-for."""
    return service.open_loops(body)


@router.post("/meeting-prep")
def meeting_prep(body: service.MeetingPrepInput, identity: Bot):
    """`context_meeting_prep`: preparation for a calendar event or the next meeting with an entity."""
    return service.meeting_prep(body)


@router.get("/changes")
def changes(identity: Bot, query: Annotated[service.ChangesInput, Query()]):
    """`context_changes`: cursor-based context changes and tombstones."""
    return service.changes(query)


@router.post("/feedback")
def feedback(body: service.FeedbackInput, identity: Bot):
    """`context_feedback`: an agent-origin correction suggestion for the owner's review in Questions."""
    return service.feedback(body)
