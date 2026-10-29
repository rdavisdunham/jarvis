"""Lists that live in two places stay in step (docs/ADDING_FEATURES.md, "Drift guards").

Each test pins one list that is maintained by hand in more than one file: the web app and the
backend, or two backend modules. A failure names the list and the files to touch. The web sources
are read as text (the backend job has no Node), so the parsers below fail loudly when a
declaration moves instead of passing on an empty match.
"""

import re
from pathlib import Path
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


# Gaps known on this branch. Each entry is a real mismatch that someone should close; shrink the
# set when it is fixed (the equality assertions below fail until you do).
SITE_ACTION_VIEW_GAPS = {"remotea"}  # Eri may name it in ui_workspace/ui_search; site-actions.ts zod rejects it.
UI_SHOW_VIEW_GAPS = {"remotea"}  # ui_show (tools.py enum and call_tool's set) cannot open the remotea page.


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
    from jarvis.tools import READ_TOOLS
    from jarvis.ui_contracts import VIEWS

    site_views = set(strings(found(r"export const views\s*=\s*\[(.*?)\]", web("site-actions.ts"), "views")))
    assert site_views <= set(VIEWS)
    assert set(VIEWS) - site_views == SITE_ACTION_VIEW_GAPS, "apps/web/src/site-actions.ts views vs VIEWS"
    show = set(READ_TOOLS["ui_show"]["parameters"]["properties"]["view"]["enum"])
    assert show <= set(VIEWS) and set(VIEWS) - show == UI_SHOW_VIEW_GAPS, "tools.py ui_show view enum"
    for tool in ("ui_search", "ui_workspace", "ui_filter"):
        assert set(READ_TOOLS[tool]["parameters"]["properties"]["view"]["enum"]) <= set(VIEWS), tool


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
