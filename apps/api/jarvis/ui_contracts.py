"""Typed device controls. No selectors, script execution or credential fields."""

VIEWS = ["today", "inbox", "week", "all", "organize", "calendar", "notes", "memory", "notifications", "settings", "reminders"]
EDITOR_KINDS = ["task", "reminder", "note", "goal", "project", "area", "space", "actor", "event", "google_event", "bulk", "memory"]


def enum(values):
    return {"type": "string", "enum": values}


def definition(description, properties, required=()):
    return {"description": description, "parameters": {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}}


UI_TOOLS = {
    "ui_activity": definition("Open or close the current device’s Activity panel. It shows durable accepted requests, questions, saved changes, Edit and safe Revert. Closing this panel does not cancel work.", {"mode": enum(["open", "close"])}, ["mode"]),
    "ui_saved_view": definition(
        "List, save, load or delete account-private named task views. Save the current Tasks tab/filter/sort/layout; list first to get exact IDs for load/delete. Never delete without a user request.",
        {"view_operation": enum(["list","save","load","delete"]), "view_name":{"type":"string","maxLength":80}, "saved_view_id":{"type":"string","maxLength":36}}, ["view_operation"]),

    "ui_state": definition("Read the current device's actual page, view options, filters, selected/visible records, editor and non-secret device preferences. Context is ephemeral; no connected device means no verified screen.", {}),
    "ui_workspace": definition(
        "Configure the current site's list/board/timeline, sorting/grouping, timeline dates, organization tab, notes mode/archive filter or Settings section. UI-only; does not edit records or dates. Read ui_state first when preserving other view choices matters.",
        {"view": enum(VIEWS), "layout": enum(["list", "board", "timeline"]),
         "sort": enum(["priority", "due", "planned", "title", "updated"]),
         "group_by": enum(["status", "project", "assignee"]),
         "timeline_date": {"type": "string", "format": "date"}, "timeline_span": {"type": "integer", "enum": [14, 30, 90]},
         "organization_tab": enum(["goal", "project", "area", "space", "actor"]),
         "settings_section": enum(["profile", "organization", "notifications", "voice", "integrations", "privacy", "system", "sharing"]),
         "notes_mode": enum(["keyword", "semantic"]), "show_archived": {"type": "boolean"}}),
    "ui_editor": definition(
        "Read or change the open record. Read first for exact field schemas and auto_save. Existing task cards auto-save each patch: a saved receipt needs no save step, and close/navigation flushes pending fields. Other forms hold drafts until save, and close refuses unsaved changes. Patch only requested fields. Discard requires an explicit user request and cancels only uncommitted fields; it cannot undo saved changes. Use domain tools when no card/form is open.",
        {"operation": enum(["read", "patch", "save", "close", "discard"]),
         "changes": {"type": "object", "maxProperties": 35, "additionalProperties": {"anyOf": [
             {"type": "string", "maxLength": 30000}, {"type": "number"}, {"type": "boolean"}, {"type": "null"},
             {"type": "array", "items": {"type": "string", "maxLength": 1000}, "maxItems": 200}]}}},
        ["operation"]),
    "ui_device": definition(
        "Change non-secret preferences on this device: Live voice, wake word or compact/comfortable density. Voices must match ui_state's allowed voices; changing voice during a session is refused. Browser permission prompts still require the user's browser interaction. Account-wide preferences use settings_update.",
        {"voice": {"type": "string", "maxLength": 50}, "wake_enabled": {"type": "boolean"},
         "density": enum(["compact", "comfortable"])}),
}

# Custom record patches have one bounded values object; no recursive payloads.
from copy import deepcopy
_patch_scalar = UI_TOOLS["ui_editor"]["parameters"]["properties"]["changes"]["additionalProperties"]
_patch_scalar["anyOf"].append({"type":"object","maxProperties":60,"propertyNames":{"maxLength":80},"additionalProperties":deepcopy(_patch_scalar)})
