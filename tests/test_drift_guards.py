"""Lists that live in two places stay in step (docs/ADDING_FEATURES.md, "Drift guards").

Each test pins one list that is maintained by hand in more than one file: the web app and the
backend, or two backend modules. A failure names the list and the files to touch. The web sources
are read as text (the backend job has no Node), so the parsers below fail loudly when a
declaration moves instead of passing on an empty match.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from typing import get_args

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps/web/src"


def web(name):
    return (WEB / name).read_text()


def strings(text):
    return re.findall(r'"([^"]+)"', text)


def found(pattern, text, what):
    match = re.search(pattern, text, re.DOTALL)
    assert match, f"Could not find {what}; update the parser in tests/test_drift_guards.py"
    return match.group(1)


def ts_union(file, name):
    return set(strings(found(rf"export type {name}\s*=([^;]+);", web(file), f"type {name} in {file}")))


def test_web_pages_match_backend_views():
    """A page the backend does not know makes every chat message from it fail validation (PR #76)."""
    from jarvis.ui_contracts import VIEWS
    from jarvis.ui_control import View

    web_views = ts_union("types.ts", "View")
    assert web_views == set(get_args(View)), "apps/web/src/types.ts View vs apps/api/jarvis/ui_control.py View"
    assert web_views == set(VIEWS), "apps/web/src/types.ts View vs apps/api/jarvis/ui_contracts.py VIEWS"
    deep_links = strings(found(r"return\s*\[(.*?)\]\s*\.includes\(view", web("task-presets.ts"), "initialView list"))
    assert set(deep_links) == web_views, "apps/web/src/task-presets.ts initialView: ?view= deep links"


def test_browser_accepts_the_views_eri_can_send():
    """ui_workspace/ui_search/ui_show may name any page; site-actions.ts zod rejected remotea after PR #76."""
    from jarvis.tools import READ_TOOLS
    from jarvis.ui_contracts import VIEWS

    site_views = set(strings(found(r"export const views\s*=\s*\[(.*?)\]", web("site-actions.ts"), "views")))
    assert site_views == set(VIEWS), "apps/web/src/site-actions.ts views vs ui_contracts.VIEWS"
    for tool in ("ui_show", "ui_search", "ui_workspace"):
        assert set(READ_TOOLS[tool]["parameters"]["properties"]["view"]["enum"]) == set(VIEWS), tool
    # ui_filter is a deliberate subset: only pages that show filterable records.
    assert set(READ_TOOLS["ui_filter"]["parameters"]["properties"]["view"]["enum"]) <= set(VIEWS)


async def test_ui_show_opens_every_page_in_its_enum():
    """call_tool once kept its own copy of the ui_show page set; it now checks the tool's enum."""
    from jarvis.domain import DomainError
    from jarvis.tools import READ_TOOLS, call_tool

    source = (ROOT / "apps/api/jarvis/tools.py").read_text()
    branch = found(r'if name == "ui_show":(.*?)\n    if name ==', source, "call_tool ui_show branch")
    assert 'READ_TOOLS["ui_show"]["parameters"]["properties"]["view"]["enum"]' in branch
    assert not re.search(r'\{\s*"organize",', branch), "call_tool ui_show: derive the page set from the enum"
    for view in READ_TOOLS["ui_show"]["parameters"]["properties"]["view"]["enum"]:
        if view == "remotea":
            continue  # Gated on the remotea setting; tests/test_ui_control.py covers it.
        assert (await call_tool("davin", "turn", 0, "ui_show", {"view": view}))["ui_action"]["view"] == view
    try:
        await call_tool("davin", "turn", 0, "ui_show", {"view": "elsewhere"})
    except DomainError as error:
        assert error.code == "INVALID_ARGUMENT"
    else:
        raise AssertionError("ui_show opened an unknown page")


def test_push_and_in_app_notifications_open_the_same_place():
    """worker.notification_url (push) and notice-target.ts (the Notifications page's Open) share the cases in
    apps/web/src/notice-targets.json; notice-target.test.ts checks the web side against the same rows."""
    from jarvis.worker import notification_url

    cases = json.loads((WEB / "notice-targets.json").read_text())
    for case in cases:
        row = case["notification"]
        url = notification_url(SimpleNamespace(task_id=row["task_id"], target=row["target"]))
        assert url == case["url"], case["name"]
        # The URL and the in-app destination name the same place.
        query = {k: v[0] for k, v in parse_qs(urlsplit(url).query, keep_blank_values=True).items()}
        opened = case["open"]
        expected = {
            "run": {"view": "remotea", "run": opened.get("run") or ""},
            "task": {"view": "tasks", "tab": "all", "record": "task:" + str(opened.get("task")),
                     "workspace": "personal"},
            "activity": {"view": "tasks", "activity": "1"},
            "today": {"view": "tasks", "tab": "today"},
            "notifications": {"view": "notifications"},
        }[opened["open"]]
        assert query == expected, case["name"]
    # Every target.view a notification is created with has a case.
    creators = "".join((ROOT / "apps/api/jarvis" / f).read_text() for f in ("notices.py", "run_coordinator.py"))
    views = set(re.findall(r'target=\{\s*"view":\s*"([a-z_]+)"', creators))
    assert views, "notification target parser"
    assert views <= {(c["notification"]["target"] or {}).get("view") for c in cases}, views
    # The page's Open button goes through the shared helper.
    button = found(r'className="notice-primary" onClick=\{(.*?)\}>Open<', web("App.tsx"), "Notifications Open button")
    assert "noticeDestination(n)" in button


def test_editor_kinds_match_the_browser():
    from jarvis.ui_contracts import EDITOR_KINDS
    from jarvis.ui_control import EditorContext

    site = strings(found(r"export const editorKinds\s*=\s*\[(.*?)\]", web("site-actions.ts"), "editorKinds"))
    assert set(site) == set(EDITOR_KINDS), "apps/web/src/site-actions.ts editorKinds vs ui_contracts.EDITOR_KINDS"
    assert set(EDITOR_KINDS) <= set(get_args(EditorContext.model_fields["kind"].annotation))


def test_settings_sections_match_everywhere():
    from jarvis.ui_contracts import UI_TOOLS
    from jarvis.ui_control import UIContext

    sections = set(re.findall(r'\{id: "([a-z_]+)"', web("settings-sections.ts")))
    assert sections, "settings-sections.ts parser"
    assert sections == set(get_args(UIContext.model_fields["settings_section"].annotation)), "ui_control.UIContext"
    tool = UI_TOOLS["ui_workspace"]["parameters"]["properties"]["settings_section"]["enum"]
    assert sections == set(tool), "ui_contracts.py ui_workspace settings_section"
    site = strings(found(r"settings_section:\s*z\s*\.enum\(\[(.*?)\]", web("site-actions.ts"), "settings_section"))
    assert sections == set(site), "apps/web/src/site-actions.ts settings_section"


def test_agent_docs_name_real_settings_sections():
    """EXTERNAL_AGENTS.md said "Settings → Integrations" after the section was relabelled "Connections"."""
    labels = re.findall(r'label: "([^"]+)"', web("settings-sections.ts"))
    assert labels, "settings-sections.ts label parser"
    for doc in ("EXTERNAL_AGENTS.md", "AGENT_CONTROL.md"):
        text = (ROOT / "docs" / doc).read_text()
        for match in re.finditer(r"Settings → ", text):
            rest = text[match.end():]
            assert any(rest.startswith(label) for label in labels), (doc, rest[:40])


def test_web_run_states_match_the_coordinator():
    from jarvis import run_coordinator
    from jarvis.session_models import TERMINAL_RUN_STATES

    assert ts_union("remotea-model.ts", "RunState") == set(run_coordinator.STATES), (
        "apps/web/src/remotea-model.ts RunState vs run_coordinator.TRANSITIONS")
    terminal = strings(found(r"export const TERMINAL_RUN_STATES[^=]*=\s*\[(.*?)\]", web("remotea-model.ts"), "terminal"))
    assert set(terminal) == set(TERMINAL_RUN_STATES)


def test_every_needs_you_reason_has_text():
    """REASONS is what the run page, push body and control_status show; a missing key shows the raw code."""
    from jarvis.run_coordinator import REASONS

    source = (ROOT / "apps/api/jarvis/run_coordinator.py").read_text()
    used = set(re.findall(r'transition\(db, run, "needs_you",[^)\n]*?reason="([a-z_]+)"', source))
    assert used, "run_coordinator.py parser"
    assert used <= set(REASONS), sorted(used - set(REASONS))


def test_every_control_tool_is_gated_on_remotea():
    """tools.py adds every control.* command to VOICE_MUTATIONS; tool_policy.CONTROL_TOOLS is what is dropped
    while remotea is off. A new control command missing from CONTROL_TOOLS would reach Eri with remotea off."""
    from jarvis.tool_policy import CONTROL_TOOLS
    from jarvis.tools import registry

    assert {d["name"] for d in registry() if d["name"].startswith("control_")} == set(CONTROL_TOOLS)


def test_settings_can_grant_every_bot_scope_but_the_link_key():
    """The key dialog builds scopes by hand (PR #72 added agents:session). agents:control is minted only by
    POST /api/v1/remotea/enable."""
    from jarvis.bot_access import SCOPES

    source = web("BotSettings.tsx")
    groups = set(re.findall(r'\["([a-z]+)","[^"]*"\]', source))
    assert groups, "BotSettings.tsx permission groups parser"
    extra = set(strings(found(r"export function permissionScopes(.*?)\n}", source, "permissionScopes")))
    grantable = {f"{g}:{level}" for g in groups for level in ("read", "write")} | {s for s in extra if ":" in s}
    assert grantable <= set(SCOPES), sorted(grantable - set(SCOPES))
    assert set(SCOPES) - grantable == {"agents:control"}, sorted(set(SCOPES) - grantable)


def test_external_record_kinds_and_mcp_reads_agree():
    from jarvis import bot_access
    from jarvis import external_service as service
    from jarvis.external_mcp import definitions

    assert set(get_args(service.Kind)) == set(service.KINDS), "external_service.Kind vs KINDS"
    for name in service.CUSTOM_READS:
        scope = bot_access.READ_SCOPES[name]
        assert name in {d["name"] for d in definitions({scope})}, (name, scope)


def test_run_dialog_harnesses_match_the_backend():
    """The Run dialog's agent choices (remotea-model.ts HARNESSES) are a copy of agent_control.HARNESSES: same
    harness ids, and the first model listed for each is the backend's default (startBody omits it then)."""
    from jarvis.agent_control import HARNESSES

    source = web("remotea-model.ts")
    block = found(r"export const HARNESSES = \[(.*?)\] as const;", source, "HARNESSES")
    entries = re.findall(r'id: "([a-z]+)", label: "[^"]+", models: \[\{ id: "([^"]+)"', block)
    assert dict(entries) == HARNESSES, "apps/web/src/remotea-model.ts HARNESSES vs apps/api/jarvis/agent_control.py HARNESSES"
