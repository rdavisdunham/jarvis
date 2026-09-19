"""Map exposed operations to owning feature areas; this is inventory, not assertion coverage."""

import ast
import json

from .catalog import CATALOG, ROOT

PREFIX = {
    "actor": "organization",
    "area": "organization",
    "space": "organization",
    "goal": "organization",
    "project": "organization",
    "calendar": "google_sync",
    "linear": "linear_sync",
    "memory": "memory_management",
    "note": "notes",
    "notelist": "note_lists",
    "notification": "notifications",
    "planning": "planning",
    "record": "custom_records",
    "routing": "routing_rules",
    "schedule": "routines",
    "settings": "browser_settings",
    "structure": "custom_schema",
    "task": "task_edit",
    "search": "search",
    "time": "time_deadlines",
    "ui": "site_controls",
    "work": "queue",
    "organization": "organization",
}
EXACT = {
    "task.create": "task_capture",
    "task.complete": "task_lifecycle",
    "task.reopen": "task_lifecycle",
    "task.batch": "task_lifecycle",
    "task.selection_update": "task_lifecycle",
    "note.tasks": "note_tasks",
    "note.organize": "note_organization",
    "note.file": "note_lists",
    "memory.capture": "memory_capture",
    "memory.resolve": "memory_dream",
    "memory_review_list": "memory_dream",
    "routing.run": "routing_dream",
    "routing.answer": "routing_dream",
    "note_extract": "note_tasks",
    "note_lists": "note_lists",
    "note_list_items": "note_lists",
    "record_search": "search",
    "search_feedback": "search_aliases",
    "search_select": "search_aliases",
    "work_revert": "receipts",
    "ui_activity": "receipts",
}
for prefix in ("calendar", "linear"):
    for action in ("create", "delete", "publish", "resolve", "update"):
        EXACT[f"{prefix}.{action}"] = prefix.replace("calendar", "google") + "_writes"


def owner(name):
    if name in EXACT:
        return EXACT[name]
    prefix = name.split(".")[0].split("_")[0]
    if prefix not in PREFIX:
        raise ValueError("New exposed tool needs a feature owner: " + name)
    return PREFIX[prefix]


def route_owner(path, source):
    if "/auth/" in path or path.endswith("/bootstrap"):
        return "accounts"
    if "/external/" in path or "/bot-keys" in path:
        return "external_agents"
    if "/accounts" in path:
        return "workspaces"
    if "/voice/" in path:
        return "live_voice"
    if "/note-lists" in path:
        return "note_lists"
    if "/search" in path:
        return "search_aliases" if any(x in path for x in ("alias", "feedback", "selection")) else "search"
    if "/structure/routing" in path:
        return "routing_rules"
    if "/structure" in path:
        return "custom_records" if "record" in path or "by-core" in path else "custom_schema"
    if "/work/actions" in path or "/commands/" in path:
        return "receipts"
    if "/work" in path:
        return "queue"
    if "/calendar" in path or "/google" in path:
        return "google_writes" if "writes" in path else "google_sync"
    if "/linear" in path:
        return "linear_writes" if "writes" in path or "comparison" in path else "linear_sync"
    if "/memory" in path:
        return "memory_dream" if "review" in path else "memory_management"
    if "/notes" in path:
        return "note_tasks" if "suggest-tasks" in path else "notes"
    if "/notifications" in path or "/push" in path:
        return "notifications"
    if "/task-views" in path:
        return "planner_views"
    if "/tasks" in path:
        return "task_edit"
    if "/schedules" in path:
        return "routines"
    if "/planning" in path:
        return "planning"
    if "/organization" in path or "/projects" in path:
        return "organization"
    if "/chat" in path or "/conversations" in path:
        return "chat_agent"
    if "/ui/" in path:
        return "site_controls"
    if "/export" in path or "/sources" in path:
        return "privacy"
    if "/budget" in path:
        return "budget"
    if "/health/" in path or "/events" in path or "/jobs" in path:
        return "operations"
    if path == "/api/v1/commands":
        return "receipts"
    if path == "/{path:path}":
        return "browser_settings"
    raise ValueError("New API route needs a feature owner: " + source + " " + path)


def routes():
    items = []
    modules = {f.stem: (f, ast.parse(f.read_text())) for f in sorted((ROOT / "apps/api/jarvis").glob("*.py"))}
    api_tree = modules["api"][1]
    aliases = {}
    for node in api_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                aliases[name.asname or name.name] = node.module
    mounted = {}
    for node in ast.walk(api_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "include_router"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            prefix = next(
                (
                    kw.value.value
                    for kw in node.keywords
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant)
                ),
                "",
            )
            mounted[aliases.get(node.args[0].id, "")] = prefix
    for module, (file, tree) in modules.items():
        prefixes = {"app": ""}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "APIRouter"
            ):
                prefix = next(
                    (
                        kw.value.value
                        for kw in node.value.keywords
                        if kw.arg == "prefix" and isinstance(kw.value, ast.Constant)
                    ),
                    "",
                )
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        prefixes[target.id] = mounted.get(module, "") + prefix
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Call)
                    and isinstance(deco.func, ast.Attribute)
                    and deco.func.attr in {"get", "post", "put", "patch", "delete", "websocket"}
                    and isinstance(deco.func.value, ast.Name)
                    and deco.func.value.id in prefixes
                    and deco.args
                    and isinstance(deco.args[0], ast.Constant)
                    and isinstance(deco.args[0].value, str)
                ):
                    path = prefixes[deco.func.value.id] + deco.args[0].value
                    source = file.relative_to(ROOT).as_posix()
                    items.append(
                        {
                            "method": deco.func.attr.upper(),
                            "path": path,
                            "handler": node.name,
                            "source": source,
                            "feature": route_owner(path, source),
                        }
                    )
    return sorted(items, key=lambda x: (x["path"], x["method"]))


def build():
    from jarvis.domain import COMMANDS
    from jarvis.tools import READ_TOOLS

    return {
        "meaning": "Surface ownership map, not proof every endpoint has an executed assertion.",
        "commands": [{"name": name, "feature": owner(name)} for name in sorted(COMMANDS)],
        "read_tools": [{"name": name, "feature": owner(name)} for name in sorted(READ_TOOLS)],
        "loop_controls": [
            {"name": "tools_load", "feature": "chat_agent"},
            {"name": "work_needs_input", "feature": "clarifications"},
            {"name": "work_answer", "feature": "clarifications"},
            {"name": "work_followup", "feature": "queue"},
            {"name": "voice_end", "feature": "wake_shutdown"},
        ],
        "http_routes": routes(),
        "mounted_services": [
            {
                "path": "/api/v1/external/mcp",
                "feature": "external_agents",
                "note": "Mounted in api.py via external_mcp.",
            }
        ],
    }


def check():
    expected = json.loads((CATALOG / "surface-inventory.json").read_text())
    actual = build()
    if expected != actual:
        raise ValueError(
            "Exposed app surfaces changed. Review new/changed functions, add evals, and refresh surface-inventory.json."
        )


if __name__ == "__main__":
    path = CATALOG / "surface-inventory.json"
    data = build()
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in data.items() if isinstance(v, list)}))
