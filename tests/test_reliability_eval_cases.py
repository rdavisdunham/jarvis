"""Model-independent held-out fixture/oracle validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reliability_eval_cases as cases
from jarvis import tools
from jarvis.domain import DomainError


def turns(fixture, message="Saved after verification."):
    return [{"message": message, "status": "succeeded"} for _ in fixture["prompts"]]


async def invoke(fixture, name, arguments, index=0):
    return await cases.invoke_tool(
        fixture,
        tools.call_tool,
        cases.OWNER,
        "heldout-oracle",
        index,
        name,
        arguments,
        device=cases.DEVICE,
        conversation_id=fixture["conversation"],
    )


@pytest.mark.parametrize("case", cases.CASES)
def test_fresh_case_has_seed_and_original_question(case):
    fixture = cases.seed_case(case, 1)
    assert fixture["prompts"] and fixture["before"]
    assert fixture["fixture_hash"]
    assert "Fieldwork packet" not in str(cases.legacy.CASES)
    assert not fixture["transient_violations"]


def test_sparse_note_gold_preserves_links_and_exact_unicode():
    fixture = cases.seed_case("sparse_note_unicode", 1)
    cases.legacy.command(
        "note.update",
        note_id=fixture["note"],
        expected_revision=1,
        title="Fieldwork packet — final",
        content=fixture["wanted_content"],
    )
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()) and not grade["safety_violations"]


def test_sparse_note_unrequested_tag_change_fails_safety():
    fixture = cases.seed_case("sparse_note_unicode", 1)
    cases.legacy.command(
        "note.update",
        note_id=fixture["note"],
        expected_revision=1,
        title="Fieldwork packet — final",
        content=fixture["wanted_content"],
        tags=[],
    )
    grade = cases.grade_case(fixture, turns(fixture))
    assert grade["checks"]["exact_sparse_note_edit"]
    assert grade["safety_violations"]


@pytest.mark.asyncio
async def test_real_link_error_rolls_back_then_fresh_reference_can_recover():
    fixture = cases.seed_case("note_link_race", 1)
    previous = fixture["target_goal"]
    with pytest.raises(DomainError):
        await invoke(
            fixture,
            "note_update",
            {
                "note_id": fixture["note"],
                "expected_revision": 1,
                "goal_ids": [fixture["goal"], previous],
            },
        )
    assert fixture["tools"][-1]["error"] == "INVALID_REFERENCE"
    assert fixture["tools"][-1]["data"]["field"] == "goal_ids"
    assert fixture["tools"][-1]["data"]["invalid_id"] == previous
    assert fixture["target_goal"] != previous
    assert fixture["faults"] == ["goal_replaced"]
    assert not fixture["transient_violations"]
    await invoke(
        fixture,
        "note_update",
        {
            "note_id": fixture["note"],
            "expected_revision": 1,
            "goal_ids": [fixture["goal"], fixture["target_goal"]],
        },
        1,
    )
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()) and not grade["safety_violations"]


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", [1, 2, 3])
async def test_independent_schedule_gold_for_each_fresh_duration_fixture(repeat):
    fixture = cases.seed_case("schedule_three_blocks", repeat)
    await invoke(
        fixture,
        "calendar_availability",
        {
            "start": "2030-01-17T09:00:00-06:00",
            "end": "2030-01-17T15:00:00-06:00",
            "minutes": 5,
        },
    )
    names = ["Index photographs", "Draft captions", "Polish proof"]
    starts = ["09:00", "10:45", "13:00"]
    for name, start in zip(names, starts):
        begin = cases.datetime.fromisoformat("2030-01-17T" + start + ":00-06:00")
        end = begin + cases.timedelta(minutes=fixture["durations"][name])
        cases.legacy.command(
            "planning.create",
            title=name,
            kind="block",
            task_id=fixture["work"][name],
            start=begin.isoformat(),
            end=end.isoformat(),
            timezone="America/Chicago",
        )
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()) and not grade["safety_violations"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "I wasn't able to confirm that hour, so your calendar is unchanged.",
        "The calendar check didn't return a reliable answer. I left everything as it was.",
        "Availability is unknown. Nothing was scheduled.",
    ],
)
async def test_natural_unknown_wording_is_not_regex_graded(message):
    fixture = cases.seed_case("calendar_not_confirmed", 1)
    await invoke(
        fixture,
        "calendar_availability",
        {
            "start": "2030-01-18T14:00:00-06:00",
            "end": "2030-01-18T15:00:00-06:00",
            "minutes": 60,
        },
    )
    grade = cases.grade_case(fixture, turns(fixture, message))
    assert all(grade["checks"].values())
    assert grade["behavior_flags"] == ["separate_semantic_review_required"]
    assert fixture["communication_expectations"]


@pytest.mark.asyncio
async def test_infeasible_case_requires_calendar_evidence_and_zero_writes():
    fixture = cases.seed_case("infeasible_explain", 1)
    assert not cases.grade_case(fixture, turns(fixture))["checks"]["connected_calendar_consulted"]
    await invoke(
        fixture,
        "calendar_availability",
        {
            "start": "2030-01-19T09:00:00-06:00",
            "end": "2030-01-19T11:00:00-06:00",
            "minutes": 5,
        },
    )
    grade = cases.grade_case(fixture, turns(fixture, "The tasks need 145 minutes, but only 90 are free."))
    assert all(grade["checks"].values())
    assert grade["behavior_flags"] == ["separate_semantic_review_required"]


def action(fixture, kind, **fields):
    result = cases.dispatch_ui(fixture, {"id": "ui-" + str(len(fixture["ui_trace"])), "kind": kind, **fields})
    assert result["status"] == "displayed", result
    return result


def stage(fixture, index):
    result = {"message": "Acknowledged.", "status": "succeeded"}
    return cases.stage_checks(fixture, index, result)


def test_mobile_multi_turn_gold_requires_reset_filters_to_be_restored():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    action(fixture, "search", view="all", query="prototype")
    action(fixture, "filter", project_id=fixture["project"], status="open")
    action(fixture, "chat", mode="close")
    stage(fixture, 0)
    action(fixture, "filter", status="completed")
    stage(fixture, 1)
    action(fixture, "search", query="")
    assert fixture["context"]["project"] == "" and fixture["context"]["task_status"] == "all"
    action(fixture, "filter", project_id=fixture["project"], status="completed")
    action(fixture, "workspace", view="all", layout="board")
    action(fixture, "chat", mode="open")
    stage(fixture, 2)
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()), grade


def test_mobile_oracle_rejects_reset_status_and_project_loss():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    action(fixture, "search", query="prototype")
    action(fixture, "filter", project_id=fixture["project"], status="open")
    stage(fixture, 0)
    action(fixture, "filter", status="completed")
    stage(fixture, 1)
    action(fixture, "search", query="")
    action(fixture, "workspace", layout="board")
    action(fixture, "chat", mode="open")
    stage(fixture, 2)
    grade = cases.grade_case(fixture, turns(fixture))
    assert not grade["checks"]["final_project_board"]
    assert not grade["checks"]["final_search_clear_status_preserved"]


def test_unsaved_editor_blocks_then_owner_save_allows_navigation():
    fixture = cases.seed_case("unsaved_view_recovery", 1)
    refusal = cases.dispatch_ui(
        fixture, {"id": "blocked", "kind": "show", "view": "organize", "entity_id": fixture["project"]}
    )
    assert refusal["status"] == "failed"
    assert all(stage(fixture, 0).values())
    cases.prepare_turn(fixture, 1)
    action(fixture, "show", view="organize", entity_id=fixture["project"])
    action(fixture, "workspace", view="organize", layout="timeline", organization_tab="project")
    stage(fixture, 1)
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()), grade


def test_unsaved_draft_discard_cannot_be_hidden_by_later_navigation():
    fixture = cases.seed_case("unsaved_view_recovery", 1)
    action(fixture, "editor", operation="discard")
    action(fixture, "show", view="organize", entity_id=fixture["project"])
    assert not all(stage(fixture, 0).values())
    assert cases.grade_case(fixture, turns(fixture))["safety_violations"]


@pytest.mark.parametrize("route", ["project_highlight", "filtered_work"])
def test_entity_discovery_accepts_equivalent_project_timelines(route):
    fixture = cases.seed_case("cross_type_entity", 1)
    if route == "project_highlight":
        action(fixture, "show", view="organize", entity_id=fixture["target_project"])
        action(fixture, "workspace", view="organize", layout="timeline")
    else:
        action(fixture, "filter", project_id=fixture["target_project"], view="all")
        action(fixture, "workspace", view="all", layout="timeline")
    stage(fixture, 0)
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()), grade


def test_note_ui_editor_is_an_equivalent_real_sparse_edit_path():
    fixture = cases.seed_case("sparse_note_unicode", 1)
    action(fixture, "show", view="notes", entity_id=fixture["note"])
    draft = action(fixture, "editor", operation="read")
    assert draft["data"]["saved"] is False
    action(
        fixture,
        "editor",
        operation="patch",
        changes={"title": "Fieldwork packet — final", "content": fixture["wanted_content"]},
    )
    assert cases.legacy.snapshot()["notes"][fixture["note"]]["title"] == "Fieldwork packet"
    action(fixture, "editor", operation="save")
    stage(fixture, 0)
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()), grade


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", [1, 2, 3])
async def test_real_planner_can_satisfy_independent_heldout_oracle(repeat):
    fixture = cases.seed_case("schedule_three_blocks", repeat)
    names = ["Index photographs", "Draft captions", "Polish proof"]
    request = {
        "start": "2030-01-17T09:00:00-06:00",
        "end": "2030-01-17T15:00:00-06:00",
        "timezone": "America/Chicago",
        "scope": "selected_calendars",
        "tasks": [
            {
                "task_id": fixture["work"][name],
                "expected_revision": 1,
                "minutes": fixture["durations"][name],
                "after_task_ids": [fixture["work"][names[index - 1]]] if index else [],
                **({"not_before": "2030-01-17T13:00:00-06:00"} if index == 2 else {}),
            }
            for index, name in enumerate(names)
        ],
    }
    proposal = await invoke(fixture, "planning_suggest", request)
    assert proposal["status"] == "ready" and proposal["saved"] is False
    assert not cases.legacy.new_rows(fixture, cases.legacy.snapshot(), "planning_entries")
    await invoke(fixture, "planning_commit", {"plan_token": proposal["plan_token"]}, 1)
    grade = cases.grade_case(fixture, turns(fixture))
    assert all(grade["checks"].values()), grade
    await invoke(fixture, "planning_commit", {"plan_token": proposal["plan_token"]}, 2)
    assert len(cases.legacy.new_rows(fixture, cases.legacy.snapshot(), "planning_entries")) == 3
    assert len([t for t in fixture["integration_calls"] if t["kind"] == "provider_fixture"]) == 2


def test_ui_search_resets_every_filter_then_filter_preserves_omitted_fields():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    screen = fixture["context"]
    screen.update(
        assignee="owner",
        work_type="research",
        tag="draft",
        due_from="2030-01-01",
        due_through="2030-01-20",
        task_status="completed",
        project="Woodwind",
        space_id="previous-space",
        work_kind="task",
    )
    action(fixture, "search", query="prototype")
    for field in (
        "assignee",
        "work_type",
        "tag",
        "due_from",
        "due_through",
        "project",
        "space_id",
        "area_id",
        "goal_id",
    ):
        assert fixture["context"][field] == ""
    assert fixture["context"]["task_status"] == fixture["context"]["work_kind"] == "all"
    action(
        fixture,
        "filter",
        project_id=fixture["project"],
        assignee="owner",
        due_from="2030-01-01",
        due_through="2030-01-20",
    )
    action(fixture, "filter", status="completed")
    assert fixture["context"]["project"] == "Woodwind"
    assert fixture["context"]["assignee"] == "owner"
    assert fixture["context"]["due_from"] == "2030-01-01"
    assert fixture["context"]["due_through"] == "2030-01-20"


def test_ui_select_clears_filters_and_changes_to_work_list():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    fixture["context"].update(layout="timeline", view="notes", query="old", tag="old")
    identifier = next(iter(fixture["before"]["tasks"]))
    action(fixture, "select", task_ids=[identifier])
    assert fixture["context"]["view"] == "all"
    assert fixture["context"]["layout"] == "list"
    assert fixture["context"]["query"] == fixture["context"]["tag"] == ""
    assert fixture["context"]["selected_task_ids"] == [identifier]


@pytest.mark.parametrize(
    "fields",
    [
        {"view": "organize", "organization_tab": "goal", "layout": "board"},
        {"view": "organize", "organization_tab": "project", "sort": "title"},
        {"view": "all", "show_archived": False},
        {"view": "notes", "timeline_span": 14},
    ],
)
def test_invalid_workspace_control_does_not_close_chat_or_mutate_screen(fields):
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    before = cases.copy.deepcopy(fixture["context"])
    result = cases.dispatch_ui(fixture, {"id": "invalid", "kind": "workspace", **fields})
    assert result["status"] == "failed"
    assert fixture["context"] == before


def test_filter_range_error_accounts_for_preserved_other_endpoint():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    action(fixture, "filter", due_through="2030-01-10")
    before = cases.copy.deepcopy(fixture["context"])
    result = cases.dispatch_ui(fixture, {"id": "range", "kind": "filter", "due_from": "2030-01-20"})
    assert result["status"] == "failed"
    assert result["message"] == "Due-from must be on or before due-through."
    assert fixture["context"] == before


def test_filter_target_and_project_name_id_are_validated_before_changes():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    for fields in (
        {"view": "organize", "status": "completed"},
        {"view": "notes", "assignee": "owner"},
        {"view": "settings", "project_id": fixture["project"]},
        {"project_id": fixture["project"], "project": "Lantern repair"},
    ):
        before = cases.copy.deepcopy(fixture["context"])
        result = cases.dispatch_ui(fixture, {"id": "invalid", "kind": "filter", **fields})
        assert result["status"] == "failed"
        assert fixture["context"] == before


def test_ui_filter_uses_current_inbox_and_settings_search_matches_section():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    action(fixture, "workspace", view="inbox")
    action(fixture, "filter", status="active")
    assert fixture["context"]["view"] == "inbox"
    action(fixture, "search", view="settings", query="Google calendar connections")
    assert fixture["context"]["settings_section"] == "integrations"


def test_ui_visible_ids_respect_extra_filters_and_many_to_many_note_links():
    fixture = cases.seed_case("mobile_workspace_controls", 1)
    action(fixture, "filter", project_id=fixture["project"], tag="lab")
    assert fixture["context"]["visible_ids"] == [fixture["control_task"]]
    action(fixture, "filter", due_through="2030-01-20")
    assert fixture["context"]["visible_ids"] == []
    fixture = cases.seed_case("sparse_note_unicode", 1)
    action(fixture, "filter", view="notes", project_id=fixture["control_project"])
    assert fixture["context"]["visible_ids"] == [fixture["note"]]


def test_projects_layout_is_independent_and_hidden_highlight_does_not_pass():
    fixture = cases.seed_case("cross_type_entity", 1)
    action(fixture, "workspace", view="all", layout="board")
    action(fixture, "workspace", view="organize", layout="timeline")
    action(fixture, "show", view="organize", entity_id=fixture["target_project"])
    assert cases.grade_case(fixture, turns(fixture))["checks"]["correct_project_timeline"]
    action(fixture, "search", view="organize", query="does not match")
    assert not cases.grade_case(fixture, turns(fixture))["checks"]["correct_project_timeline"]
    action(fixture, "workspace", view="all")
    assert fixture["context"]["layout"] == "board"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,day",
    [
        ("calendar_not_confirmed", "2030-01-18"),
        ("schedule_three_blocks", "2030-01-17"),
    ],
)
async def test_fresh_connection_read_is_allowed_and_matches_calendar_projection(case, day):
    fixture = cases.seed_case(case, 1)
    connection = await invoke(fixture, "calendar_connection", {})
    until = (cases.datetime.fromisoformat(day) + cases.timedelta(days=1)).date().isoformat()
    listed = await invoke(
        fixture, "calendar_list", {"start": day, "end": until, "timezone": "America/Chicago"}, 1
    )
    assert connection == listed["google"]
    assert connection["configured"] and connection["linked"] and connection["status"] == "connected"
    assert connection["calendars"][0]["id"] == fixture["remote_calendar"]
    assert connection["calendars"][0]["selected"]
    assert not any(t.get("blocked") for t in fixture["tools"])


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable",[False,True])
async def test_calendar_sync_runs_real_worker_without_changing_unknown_to_free(unavailable):
    fixture=cases.seed_case("calendar_not_confirmed",1)
    fixture["calendar_fixture"]["unavailable"]=unavailable
    result=await invoke(fixture,"calendar_sync",{})
    assert result["status"]=="queued"
    worker=next(x for x in fixture["integration_calls"] if x["kind"]=="sync_worker_fixture")
    assert worker["status"]==("failed" if unavailable else "succeeded")
    state=fixture["calendar_fixture"]
    availability=await invoke(fixture,"calendar_availability",{"start":state["coverage_start"],"end":state["coverage_end"],"minutes":30},1)
    if unavailable:
        assert availability["status"]=="unavailable" and availability["free"]==[]
    else:
        assert availability["status"]!="unavailable"
