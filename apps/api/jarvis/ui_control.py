"""Device-local UI bridge. Context and acknowledgements are ephemeral, never conversation memory."""

import asyncio
import json
import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class EditorContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["task", "reminder", "note", "goal", "project", "area", "space", "actor", "event", "google_event", "bulk", "memory"]
    record_id: str | None = Field(default=None, max_length=36)
    dirty: bool = False
    busy: bool = False
    fields: list[Annotated[str, Field(max_length=80)]] = Field(default_factory=list, max_length=35)


class DevicePreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice: str = Field(default="", max_length=50)
    voices: list[Annotated[str, Field(max_length=50)]] = Field(default_factory=list, max_length=30)
    wake_enabled: bool = False
    wake_supported: bool = False
    density: Literal["compact", "comfortable"] = "compact"
    private_chat: bool = False


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
    task_status: Literal["all", "active", "open", "in_progress", "waiting", "deferred", "completed", "cancelled"] = (
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
    layout: Literal["list", "board", "timeline"] = "list"
    sort: Literal["priority", "due", "planned", "title", "updated"] = "priority"
    group_by: Literal["status", "project", "assignee"] = "status"
    timeline_date: str = Field(default="", max_length=10)
    timeline_span: Literal[14, 30, 90] = 30
    organization_tab: Literal["goal", "project", "area", "space", "actor"] = "project"
    settings_section: Literal["profile", "voice", "integrations", "privacy", "system"] = "profile"
    notes_mode: Literal["keyword", "semantic"] = "keyword"
    show_archived: bool = False
    assignee: str = Field(default="", max_length=100)
    work_type: str = Field(default="", max_length=80)
    tag: str = Field(default="", max_length=40)
    due_from: str = Field(default="", max_length=10)
    due_through: str = Field(default="", max_length=10)
    editor: EditorContext | None = None
    device_preferences: DevicePreferences = Field(default_factory=DevicePreferences)



class UISync(BaseModel):
    context: UIContext
    results: list[dict] = Field(default_factory=list, max_length=30)

    @field_validator("results")
    @classmethod
    def bounded_results(cls, values):
        for item in values:
            if set(item) - {"id", "status", "message", "data"} or not isinstance(item.get("id"), str) or len(item["id"]) > 150:
                raise ValueError("Invalid UI acknowledgement.")
            if item.get("status") not in {"displayed", "failed"} or len(str(item.get("message", ""))) > 1000:
                raise ValueError("Invalid UI outcome.")
            if len(json.dumps(item, ensure_ascii=False)) > 150000:
                raise ValueError("UI outcome is too large.")
        return values


states = {}
pending = {}

APP_MAP = """Site map:
Today, Inbox (unclassified) and This week (next seven days) are presets of Work.
Work (all): one task store with list, status/project/assignee boards, and dated timelines.
Projects & goals (organize): Projects list/board/start-target timeline; separate Goals, Areas,
Spaces and People & agents tabs. Goal metrics are outcomes, project counts are task completion.
Calendar: month/week/day, task planned/deadline markers, alerts, local appointments/work blocks and Google events.
A planned date/deadline does not reserve time; timeline task markers are not duration bars.
Notes: authored content, home project, related goals/projects/notes, backlinks and task evidence.
Memory: learned facts and review questions. Notifications: delivered task alerts.
Settings sections: profile (name/reminder defaults/density), voice (Live voice/wake word),
integrations (Google/Linear), privacy (history/learning), system (backend model/usage/backups/export).
Realtime is disabled. Browser permissions, OAuth consent and credentials need the owner's interaction.
Filters & sort expands from one control; collapsed filters still apply. Layout changes preserve them.
Chat is a desktop side panel/mobile overlay. Closing chat keeps voice running; show mobile content unobscured.
Editors expose typed drafts through ui_editor; reading/filling is not saving. Never discard an unsaved edit
without an explicit owner request. Remote jobs remain pending until confirmed; local saves are distinct.
Use the planner group for constrained multi-task scheduling. It verifies a single-person schedule against
fresh availability and atomically saves local blocks only when asked. No automatic Google publication.
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
                        "data": result.get("data"),
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
        if action["kind"] in {"editor", "device"}:
            return {"status": "failed", "message": "No authenticated device is available for this action."}
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
