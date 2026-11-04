"""The release gates of plan_final.md section 11 as data, and the rules that grade them.

- Mechanical gates are suite-level: each names the tests that cover it and is graded from a JUnit XML file of a
  test run (`pytest --junitxml`), plus the evaluation's own probes where it has one. Without test results a gate is
  "not run", never "pass".
- Quality targets are graded from verified labels only, with the sample count and a 95% Wilson interval. With no
  verified sample a target is "not gradeable"; the unverified estimate is shown beside it for information.
- Critical temporal and permission scenarios fail the release on their own: averages cannot override them. A
  failure in mechanical data (a deleted meeting's content returned, another owner's data disclosed) counts whatever
  the labels say; a failure that rests on unverified labels is listed as a warning to adjudicate.

The overall verdict is FAIL when anything fails, else NOT GRADEABLE when anything is ungraded, else PASS.
"""

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from xml.etree import ElementTree

PASS, FAIL, NOT_RUN, NOT_GRADEABLE, REPORTED = "pass", "fail", "not run", "not gradeable", "reported"
Z95 = 1.959964


@dataclass(frozen=True)
class Mechanical:
    key: str
    title: str
    tests: tuple[str, ...]  # pytest node IDs (a parametrized test matches all its cases)
    pending: str = ""  # what is not covered yet and which package covers it


MECHANICAL = (
    Mechanical("unapproved_task_writes", "Zero unapproved task writes", (
        "tests/context/test_pipeline.py::test_seeding_runs_as_the_owner_even_under_a_bot_credential",
        "tests/context/test_context_owner.py::test_context_resolve_command_is_human_from_the_browser_and_agent_from_dani",
        "tests/context/test_service.py::test_the_owner_accepts_dismisses_or_defers_agent_suggestions",
        "tests/context/test_publication.py::test_agents_cannot_approve",
        "tests/context/test_review_api.py::test_approve_publishes_once_and_agents_are_refused",
        "tests/context/test_publication.py::test_changed_payload_needs_a_new_approval_and_approved_work_is_never_modified",
        "tests/context/test_review_api.py::test_updates_need_their_own_approval",
        "tests/context/test_publication.py::test_link_approval_records_the_task_without_writing_it",
        "tests/context/test_review_api.py::test_retry_route_is_human_only",
    )),
    Mechanical("duplicate_task_writes", "Zero duplicate task writes", (
        "tests/context/test_pipeline.py::test_replaying_every_job_and_redelivering_change_nothing",
        "tests/context/test_end_to_end.py::test_fixture_history_converges_across_orders_and_repeats_are_no_ops",
        "tests/context/test_lint.py::test_ten_runs_without_new_evidence_create_nothing_and_skip_the_model",
        "tests/context/test_publication.py::test_approval_publishes_exactly_one_task_with_meeting_provenance",
        "tests/context/test_publication.py::test_crash_during_publish_and_duplicate_delivery_write_one_task",
        "tests/context/test_publication.py::test_lost_commit_replays_stored_command_results",
        "tests/context/test_publication.py::test_linear_publication_reuses_the_mapping_so_sync_never_duplicates",
        "tests/context/test_publication.py::test_remote_timeout_reconciles_under_the_same_issue_before_retrying",
        "tests/context/test_publication.py::test_unconfirmed_linear_publication_retries_with_the_original_identity",
    )),
    Mechanical("cross_owner_disclosure", "Zero cross-owner disclosures", (
        "tests/context/test_service.py::test_hits_for_other_owners_never_leak",
        "tests/context/test_retrieve.py::test_scope_holds_in_both_legs_and_never_leaks_across_owners",
        "tests/context/test_service.py::test_a_key_revoked_during_retrieval_gets_nothing",
        "tests/context/test_sources.py::test_connection_must_belong_to_the_owner",
        "tests/context/test_context_owner.py::test_session_delete_flag_and_shared_workspace",
    )),
    Mechanical("processing_accounting", "Complete processing accounting for ready sources", (
        "tests/context/test_end_to_end.py::test_fixture_history_converges_across_orders_and_repeats_are_no_ops",
        "tests/context/test_extract.py::test_long_meeting_full_coverage_and_cost_per_meeting",
        "tests/context/test_models.py::test_runs_record_epoch_and_watermark_for_staleness",
    )),
    Mechanical("surviving_provenance", "Valid surviving provenance on displayed claims", (
        "tests/context/test_apply.py::test_inexact_evidence_is_refused",
        "tests/context/test_evidence.py::test_deleting_one_of_two_sources_keeps_the_assertion_and_the_last_retracts_it",
        "tests/context/test_service.py::test_evidence_from_a_deleted_source_is_withheld",
        "tests/context/test_service.py::test_brief_keeps_only_claims_whose_citations_survive_and_needs_tasks_read_for_task_claims",
        "tests/context/test_briefs.py::test_source_deletion_stales_and_rebuild_forgets_the_deleted_statement",
        "tests/context/test_retrieve.py::test_raw_segments_are_annotated_and_stale_passages_stay_out_of_current_answers",
    )),
    Mechanical("concurrency_deletion_pause_undo_replay", "Concurrency, deletion, pause, undo and replay tests pass", (
        "tests/context/test_pipeline.py::test_a_new_revision_during_the_model_call_makes_the_result_stale",
        "tests/context/test_pipeline.py::test_pause_rejects_outstanding_results_and_resume_continues_without_new_calls",
        "tests/context/test_pipeline.py::test_deletion_job_purges_and_retracts",
        "tests/context/test_pipeline_wiring.py::test_admin_pause_rejects_results_and_schedule_due_requeues_what_is_still_current",
        "tests/context/test_admin.py::test_pause_rejects_results_of_runs_started_before_it",
        "tests/context/test_decisions.py::test_wrong_retracts_records_a_negative_and_stales_briefs_and_undo_restores",
        "tests/context/test_decisions.py::test_undo_refuses_when_a_later_decision_depends_on_it",
        "tests/context/test_decisions.py::test_decision_seq_is_allocated_per_owner_under_lock",
        "tests/context/test_sources.py::test_deletion_excludes_now_and_purge_blanks_content_but_keeps_a_tombstone",
        "tests/context/test_lint.py::test_a_pass_stops_when_the_owner_is_paused_or_the_epoch_moves",
        "tests/context/test_publication.py::test_paused_processing_still_publishes_approved_work",
        "tests/context/test_publication.py::test_update_against_a_task_edited_since_is_a_conflict_not_an_overwrite",
        "tests/context/test_review_api.py::test_stale_revision_refuses_the_whole_batch",
        "tests/context/test_pipeline_wiring.py::test_a_statement_said_again_after_a_revision_dropped_it_comes_back_unless_marked_wrong",
    )),
)


def junit_outcomes(path: str | Path) -> dict[str, str]:
    """`file.py::name[param]` -> passed | failed | skipped, from a pytest JUnit XML file."""
    outcomes = {}
    for case in ElementTree.parse(path).getroot().iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        node = f"{classname.replace('.', '/')}.py::{name}"
        if case.find("failure") is not None or case.find("error") is not None:
            outcomes[node] = "failed"
        elif case.find("skipped") is not None:
            outcomes[node] = "skipped"
        else:
            outcomes[node] = "passed"
    return outcomes


def grade_mechanical(gate: Mechanical, outcomes: dict[str, str] | None, probes: list[dict] = ()) -> dict:
    """PASS only when every listed test ran and passed and every probe held; FAIL on any failure; else NOT RUN."""
    failed_probes = [p for p in probes if not p["ok"]]
    tests = {}
    for node in gate.tests:
        if outcomes is None:
            tests[node] = "not run"
            continue
        found = [v for k, v in outcomes.items() if k == node or k.startswith(node + "[")]
        tests[node] = "failed" if "failed" in found else "missing" if not found or "passed" not in found else "passed"
    if failed_probes or "failed" in tests.values():
        status = FAIL
    elif outcomes is None or any(v != "passed" for v in tests.values()):
        status = NOT_RUN
    else:
        status = PASS
    return {"key": gate.key, "title": gate.title, "status": status, "tests": tests, "probes": list(probes),
            "pending": gate.pending}


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None without a sample."""
    if n <= 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


@dataclass
class Count:
    successes: int = 0
    n: int = 0

    def add(self, successes: int | None, n: int | None) -> "Count":
        self.successes += successes or 0
        self.n += n or 0
        return self

    @property
    def rate(self) -> float | None:
        return round(self.successes / self.n, 4) if self.n else None


@dataclass
class Quality:
    key: str
    title: str
    target: float | None  # None: reported without a threshold
    verified: Count = field(default_factory=Count)
    unverified: Count = field(default_factory=Count)  # every label, verified or not (information only)
    lower_is_better: bool = False
    note: str = ""


def grade_quality(q: Quality) -> dict:
    """Verified sample only. Without a target the metric is REPORTED; with one it passes when the verified
    estimate meets it (the interval is shown so a thin sample is visible)."""
    if not q.verified.n:
        status = NOT_GRADEABLE if q.target is not None else REPORTED
    elif q.target is None:
        status = REPORTED
    else:
        met = q.verified.rate <= q.target if q.lower_is_better else q.verified.rate >= q.target
        status = PASS if met else FAIL
    return {"key": q.key, "title": q.title, "target": q.target, "status": status, "lower_is_better": q.lower_is_better,
            "verified": {"successes": q.verified.successes, "n": q.verified.n, "rate": q.verified.rate,
                         "interval": wilson(q.verified.successes, q.verified.n)},
            "unverified": {"successes": q.unverified.successes, "n": q.unverified.n, "rate": q.unverified.rate},
            "note": q.note}


@dataclass
class Critical:
    key: str
    title: str
    kind: str  # temporal | permission
    failures: list[str] = field(default_factory=list)  # count against the release whatever the labels say
    warnings: list[str] = field(default_factory=list)  # rest on unverified labels: adjudicate first
    checked: int = 0


def grade_critical(c: Critical) -> dict:
    status = FAIL if c.failures else PASS if c.checked else NOT_GRADEABLE
    return {"key": c.key, "title": c.title, "kind": c.kind, "status": status, "checked": c.checked,
            "failures": c.failures, "warnings": c.warnings}


def verdict(mechanical: list[dict], quality: list[dict], critical: list[dict]) -> str:
    """Critical scenario failures fail the release whatever the averages; then any failure; then anything ungraded
    (no test results, no verified sample) leaves the release NOT GRADEABLE. PASS needs every gate graded."""
    if any(c["status"] == FAIL for c in critical):
        return "FAIL"
    if any(g["status"] == FAIL for g in (*mechanical, *quality)):
        return "FAIL"
    if any(g["status"] in (NOT_RUN, NOT_GRADEABLE) for g in (*mechanical, *quality, *critical)):
        return "NOT GRADEABLE"
    return "PASS"
