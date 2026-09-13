"""Typed device controls. No selectors, script execution or credential fields."""

VIEWS = ["today", "inbox", "week", "all", "organize", "calendar", "notes", "memory", "notifications", "settings", "reminders"]
EDITOR_KINDS = ["task", "reminder", "note", "goal", "project", "area", "space", "actor", "event", "google_event", "bulk", "memory"]


def enum(values):
    return {"type": "string", "enum": values}


def definition(description, properties, required=()):
    return {"description": description, "parameters": {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}}


UI_TOOLS = {
    "ui_state": definition("Read the current device's actual page, view options, filters, selected/visible records, editor and non-secret device preferences. Context is ephemeral; no connected device means no verified screen.", {}),
    "ui_workspace": definition(
        "Configure the current site's list/board/timeline, sorting/grouping, timeline dates, organization tab, notes mode/archive filter or Settings section. UI-only; does not edit records or dates. Read ui_state first when preserving other view choices matters.",
        {"view": enum(VIEWS), "layout": enum(["list", "board", "timeline"]),
         "sort": enum(["priority", "due", "planned", "title", "updated"]),
         "group_by": enum(["status", "project", "assignee"]),
         "timeline_date": {"type": "string", "format": "date"}, "timeline_span": {"type": "integer", "enum": [14, 30, 90]},
         "organization_tab": enum(["goal", "project", "area", "space", "actor"]),
         "settings_section": enum(["profile", "voice", "integrations", "privacy", "system"]),
         "notes_mode": enum(["keyword", "semantic"]), "show_archived": {"type": "boolean"}}),
    "ui_editor": definition(
        "Read, fill, save or close the currently open editor. Read first for its exact editable field schemas and current draft values. Patch only requested fields; filling a draft is NOT saving. Save awaits the normal validated application command. Close refuses unsaved changes; discard is only for an explicit owner request to abandon them. Use normal domain tools for direct record changes when no form is being edited.",
        {"operation": enum(["read", "patch", "save", "close", "discard"]),
         "changes": {"type": "object", "maxProperties": 35, "additionalProperties": {"anyOf": [
             {"type": "string", "maxLength": 30000}, {"type": "number"}, {"type": "boolean"}, {"type": "null"},
             {"type": "array", "items": {"type": "string", "maxLength": 1000}, "maxItems": 200}]}}},
        ["operation"]),
    "ui_device": definition(
        "Change non-secret preferences on this device: Live voice, wake word, compact/comfortable density or current-chat privacy. Voices must match ui_state's allowed voices; changing voice during a session is refused. Browser permission prompts still require the user's browser interaction. Account-wide preferences use settings_update.",
        {"voice": {"type": "string", "maxLength": 50}, "wake_enabled": {"type": "boolean"},
         "density": enum(["compact", "comfortable"]), "private_chat": {"type": "boolean"}}),
}
