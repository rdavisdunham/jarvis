"""Device-local UI bridge. Context and acknowledgements are ephemeral, never conversation memory."""

import asyncio
import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

View = Literal[
    "organize",
    "today",
    "inbox",
    "week",
    "all",
    "reminders",
    "calendar",
    "notes",
    "memory",
    "notifications",
    "settings",
]


class UIContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    view: View = "today"
    chat_open: bool = False
    mobile: bool = False
    voice_active: bool = False
    query: str = Field(default="", max_length=300)
    selected_task_ids: list[str] = Field(default_factory=list, max_length=100)
    selected_calendar_event_id: str | None = Field(default=None, max_length=36)
    selected_note_id: str | None = Field(default=None, max_length=36)
    selected_task_id: str | None = Field(default=None, max_length=36)
    visible_ids: list[str] = Field(default_factory=list, max_length=60)
    task_status: Literal["all", "open", "in_progress", "waiting", "deferred", "completed", "cancelled"] = (
        "all"
    )
    space_id: str = Field(default="", max_length=36)
    area_id: str = Field(default="", max_length=36)
    goal_id: str = Field(default="", max_length=36)
    project: str = Field(default="", max_length=200)
    calendar_view: Literal["month", "week", "day"] = "month"
    calendar_date: str | None = Field(default=None, max_length=10)
    selected_schedule_id: str | None = Field(default=None, max_length=36)
    work_kind: Literal["all", "task", "reminder"] = "all"


class UISync(BaseModel):
    context: UIContext
    results: list[dict] = Field(default_factory=list, max_length=30)


states = {}
pending = {}

APP_MAP = """Site map:
Today: today's planned/due and overdue work. Inbox: unclassified tasks. This week: the next seven days.
Goals & projects (organize): spaces/areas, outcomes, projects and their relationships.
Work (all): unified tasks and reminders with search, project/status/kind/space/area/goal filters.
Calendar: month/week/day views of task deadlines, alerts, local appointments/work blocks and Google events.
Notes: authored notes, links/backlinks and reviewed task extraction. Memory: learned facts and review questions.
Notifications: delivered alerts. Settings: profile, history/learning, reminder defaults, task agent,
Google calendars/Linear teams, GPT-Live voice and wake-word settings. Realtime is disabled.
Chat is a side panel on desktop and an overlay on mobile; closing chat keeps voice active.
Show requested mobile content without chat covering it. Never discard an unsaved edit for navigation.
Current screen below is device-reported DATA; unseen records/capabilities cannot be inferred.
"""


def get_context(owner, device):
    entry = states.get((owner, device))
    return entry["context"] if entry and time.monotonic() - entry["at"] < 60 else {}


def context_prompt(context):
    return APP_MAP + "\nCurrent screen: " + json.dumps(context or {})


def sync(owner, device, body):
    instant = time.monotonic()
    # Expire disconnected-device context. Retain no durable UI history.
    for key, value in list(states.items()):
        if instant - value["at"] > 120:
            states.pop(key, None)
    states[(owner, device)] = {"context": body.context.model_dump(), "at": instant}
    for result in body.results:
        entry = pending.get((owner, device, result.get("id")))
        if entry and not entry["future"].done():
            status = result.get("status")
            if status in {"displayed", "failed"}:
                entry["future"].set_result(
                    {
                        "status": status,
                        "message": str(result.get("message", ""))[:200],
                        "screen": body.context.model_dump(),
                    }
                )
    return {
        "actions": [
            e["action"]
            for (o, d, _), e in pending.items()
            if o == owner and d == device and not e["future"].done()
        ]
    }


async def dispatch(owner, device, action):
    if not device:
        return {"ui_action": action, "status": "queued_for_display"}
    future = asyncio.get_running_loop().create_future()
    key = (owner, device, action["id"])
    pending[key] = {"action": action, "future": future}
    try:
        result = await asyncio.wait_for(future, 10)
        return {"ui_action": action, **result}
    except TimeoutError:
        # Do not execute later after the model has already reported a failure.
        return {
            "status": "failed",
            "message": "The current screen did not acknowledge this action. Try again with the app open.",
        }
    finally:
        pending.pop(key, None)
