# Context platform release report (plan section 11)

Generated 2026-11-04 12:01 UTC. Verdict: **NOT GRADEABLE**.

**Unverified labels excluded: 227** (116 labels, 111 questions). Verified included: 0 labels, 0 questions, 0 adjudicated series.

Synthetic histories: alpha, beta, gamma, delivered through the real pipeline (doubles, hashed bag-of-words embeddings, no model calls) in a throwaway database. Test results: `/tmp/claude-1000/-home-davin-todue/6a047dfa-828a-42a5-8cc8-dc93bca474e8/scratchpad/full.xml`.

Rule: critical temporal and permission failures fail the release whatever the averages; a target with no verified sample is NOT GRADEABLE, never a pass.

## Mechanical gates

| Gate | Status | Tests passed | Evaluation probes | Not yet covered |
|---|---|---|---|---|
| Zero unapproved task writes | PASS | 9/9 | 3/3 held |  |
| Zero duplicate task writes | PASS | 9/9 | 3/3 held |  |
| Zero cross-owner disclosures | PASS | 5/5 | 3/3 held |  |
| Complete processing accounting for ready sources | PASS | 3/3 | 3/3 held |  |
| Valid surviving provenance on displayed claims | PASS | 6/6 | 6/6 held |  |
| Concurrency, deletion, pause, undo and replay tests pass | PASS | 14/14 | 0/0 held |  |
| p95 context_search under 2 s at the corpus fixture size | PASS | 0/0 | 7/7 held |  |

## Critical scenarios (averages cannot override)

| Scenario | Status | Checked (verified or mechanical) | Failures | Warnings on unverified labels |
|---|---|---|---|---|
| Temporal scenarios: as-of answers, supersession under three orders, stale answers from the platform arms | NOT GRADEABLE | 0 | 0 | 24 |
| Permission scenarios: deleted meetings' content, disclosure to another owner | PASS | 237 | 0 | 0 |

Temporal scenarios: as-of answers, supersession under three orders, stale answers from the platform arms:
- warning (unverified, adjudicate first) alpha: as-of q-alpha-host-sep answered stale
- warning (unverified, adjudicate first) alpha: 2 supersession errors on unverified labels (arrival)
- warning (unverified, adjudicate first) alpha: 2 supersession errors on unverified labels (chronological)
- warning (unverified, adjudicate first) alpha: 2 supersession errors on unverified labels (shuffled)
- warning (unverified, adjudicate first) beta: as-of q-beta-release-oct answered stale
- warning (unverified, adjudicate first) beta: 1 supersession errors on unverified labels (arrival)
- warning (unverified, adjudicate first) beta: 1 supersession errors on unverified labels (chronological)
- warning (unverified, adjudicate first) beta: 1 supersession errors on unverified labels (shuffled)
- warning (unverified, adjudicate first) gamma: as-of q-gamma-term-aug answered stale
- warning (unverified, adjudicate first) gamma: 1 supersession errors on unverified labels (arrival)
- warning (unverified, adjudicate first) gamma: 1 supersession errors on unverified labels (chronological)
- warning (unverified, adjudicate first) gamma: 1 supersession errors on unverified labels (shuffled)
- warning (unverified, adjudicate first) alpha: qa-03 (structured) answered with a superseded value
- warning (unverified, adjudicate first) alpha: qa-03 (combined) answered with a superseded value
- warning (unverified, adjudicate first) alpha: qa-36 (combined) answered with a superseded value
- warning (unverified, adjudicate first) alpha: qa-37 (combined) answered with a superseded value
- warning (unverified, adjudicate first) beta: qb-01 (structured) answered with a superseded value
- warning (unverified, adjudicate first) beta: qb-32 (structured) answered with a superseded value
- warning (unverified, adjudicate first) beta: qb-01 (combined) answered with a superseded value
- warning (unverified, adjudicate first) beta: qb-32 (combined) answered with a superseded value
- warning (unverified, adjudicate first) gamma: qg-05 (structured) answered with a superseded value
- warning (unverified, adjudicate first) gamma: qg-32 (structured) answered with a superseded value
- warning (unverified, adjudicate first) gamma: qg-05 (combined) answered with a superseded value
- warning (unverified, adjudicate first) gamma: qg-32 (combined) answered with a superseded value

## Quality targets (verified labels only)

| Metric | Target | Status | Verified estimate [95% CI] (successes/n) | Unverified, informational | Note |
|---|---|---|---|---|---|
| Precision of automatically applied entity links | >= 0.95 | NOT GRADEABLE | n/a (n=0) | 1.000 (6/6) |  |
| Precision of preselected proposals | >= 0.95 | NOT GRADEABLE | n/a (n=0) | 1.000 (2/2) | own-action proposals; graded once a series adjudicates them |
| Recall of labelled actionable commitments | >= 0.90 | NOT GRADEABLE | n/a (n=0) | 1.000 (21/21) |  |
| Recall of labelled decisions and rationale | >= 0.90 | NOT GRADEABLE | n/a (n=0) | 0.688 (11/16) |  |

## Reported metrics

| Metric | Verified [95% CI] | Unverified, informational | Note |
|---|---|---|---|
| Attribution errors (share of matched commitments) | n/a (n=0) | 0.000 (0/21) |  |
| Fabricated actions (hypothetical or negated statements made commitments or proposals, per labelled statement) | n/a (n=0) | 0.667 (8/12) |  |
| Stale-answer rate (as-of questions and comparison questions with a stale answer, platform arms) | n/a (n=0) | 0.395 (15/38) |  |
| Supersession errors (expected changes missed, over three orders) | n/a (n=0) | 1.000 (12/12) |  |
| Human overturn rate (adjudicated items marked wrong, changed or must-not) | n/a (n=0) | n/a (n=0) | from adjudicated series only |
| Repeated-question rate (Questions created by unchanged maintenance runs, per open Question) | n/a (n=0) | 0.000 (0/7) |  |

Correction minutes per meeting: n/a (recorded in adjudicated worksheets). Pending-review age: n/a until review and publication (D1/D2) record dispositions.

## Shadow comparison: raw history vs native structured vs combined

111 questions, 0 verified, unverified excluded from release scoring: 111.

Release comparison: NOT GRADEABLE (no verified questions). Informational, all questions:

| Arm | nDCG@5 | Recall@10 | Stale answers | Leaks | Citation validity |
|---|---|---|---|---|---|
| raw | 0.867 (n=105) | 1.000 | 6/15 (0.400) | 0/6 | 1.000 (722/722) |
| structured | 0.425 (n=105) | 0.452 | 5/15 (0.333) | 0/6 | 1.000 (1015/1015) |
| combined | 0.782 (n=105) | 0.986 | 7/15 (0.467) | 0/6 | 1.000 (1081/1081) |

| Comparison | nDCG@5 difference [95% CI] | Recall@10 difference | Stale-answer difference |
|---|---|---|---|
| structured vs raw | -0.442 [-0.534, -0.347] (n=105) | -0.548 [-0.638, -0.452] (n=105) | -0.067 [-0.200, +0.000] (n=15) |
| combined vs raw | -0.085 [-0.121, -0.049] (n=105) | -0.014 [-0.033, +0.000] (n=105) | +0.067 [+0.000, +0.200] (n=15) |
| combined vs structured | +0.357 [+0.274, +0.438] (n=105) | +0.533 [+0.438, +0.624] (n=105) | +0.133 [+0.000, +0.333] (n=15) |

On these tiny histories one chunk usually holds a whole meeting, so the raw arm's recall@10 is close to trivially high; the doubles extract only decisions and commitments, so the structured arm cannot answer fact, risk or rationale questions. Neither says anything about the native engine.

## Performance

p95 context_search under 2 s at the corpus fixture size: **PASS**. Gated at scale 1 only: every scope's p95 under 2000 ms and no leaks; a leak at any scale fails.

### Scale 1 (gated)

Corpus: context_sources 500, context_source_revisions 550, context_segments 50000, context_chunks 2757, context_decisions 217, context_assertions 7243, context_assertion_versions 8923, context_evidence 8439, context_entity_links 12751, context_owner_state 3. Leaks: 0.

| Scope | p50 (ms) | p95 (ms) | Queries | Under 2000 ms |
|---|---|---|---|---|
| entity | 26.5 | 30.2 | 200 | yes |
| entity+raw | 62.3 | 77.7 | 200 | yes |
| known_at | 100.1 | 134.7 | 200 | yes |
| owner | 41.7 | 47.8 | 200 | yes |
| owner+raw | 68.7 | 83.6 | 200 | yes |

### Scale 4 (reported, not gated)

Corpus: context_sources 2000, context_source_revisions 2200, context_segments 200000, context_chunks 11054, context_decisions 868, context_assertions 28946, context_assertion_versions 35581, context_evidence 33801, context_entity_links 50492, context_owner_state 3. Leaks: 0.

| Scope | p50 (ms) | p95 (ms) | Queries | Under 2000 ms |
|---|---|---|---|---|
| entity | 61.3 | 70.6 | 200 | yes |
| entity+raw | 144.7 | 172.7 | 200 | yes |
| known_at | 245.6 | 370.9 | 200 | yes |
| owner | 128.3 | 141.7 | 200 | yes |
| owner+raw | 189.9 | 219.6 | 200 | yes |

Synthetic histories (tiny corpus, informational): worst platform-arm p95 0.0636 s.

## Cost by meeting, stage and query

### alpha (total 0.000000 USD)

| Stage | Runs | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|
| brief | 2 | 0 | 0 | 0.000000 |
| embed | 7 | 0 | 0 | 0.000000 |
| extract | 7 | 0 | 0 | 0.000000 |
| ingest | 7 | 0 | 0 | 0.000000 |
| link | 7 | 0 | 0 | 0.000000 |
| lint | 11 | 0 | 0 | 0.000000 |
| proposals | 7 | 0 | 0 | 0.000000 |
| reconcile | 7 | 0 | 0 | 0.000000 |

| Meeting | Runs | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|
| alpha-01 | 6 | 0 | 0 | 0.000000 |
| alpha-02 | 6 | 0 | 0 | 0.000000 |
| alpha-03 | 6 | 0 | 0 | 0.000000 |
| alpha-04 | 6 | 0 | 0 | 0.000000 |
| alpha-05 | 6 | 0 | 0 | 0.000000 |
| alpha-06 | 6 | 0 | 0 | 0.000000 |
| alpha-07 | 6 | 0 | 0 | 0.000000 |

Per query (USD, total / per query): raw 0.000000 / 0.000000, structured 0.000000 / 0.000000, combined 0.000000 / 0.000000. Budget usage events by feature: none.

### beta (total 0.000000 USD)

| Stage | Runs | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|
| brief | 3 | 0 | 0 | 0.000000 |
| embed | 8 | 0 | 0 | 0.000000 |
| extract | 8 | 0 | 0 | 0.000000 |
| ingest | 8 | 0 | 0 | 0.000000 |
| link | 8 | 0 | 0 | 0.000000 |
| lint | 11 | 0 | 0 | 0.000000 |
| proposals | 8 | 0 | 0 | 0.000000 |
| reconcile | 8 | 0 | 0 | 0.000000 |

| Meeting | Runs | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|
| beta-01 | 6 | 0 | 0 | 0.000000 |
| beta-02 | 6 | 0 | 0 | 0.000000 |
| beta-03 | 6 | 0 | 0 | 0.000000 |
| beta-04 | 6 | 0 | 0 | 0.000000 |
| beta-05 | 6 | 0 | 0 | 0.000000 |
| beta-06 | 6 | 0 | 0 | 0.000000 |
| beta-07 | 6 | 0 | 0 | 0.000000 |
| beta-08 | 6 | 0 | 0 | 0.000000 |

Per query (USD, total / per query): raw 0.000000 / 0.000000, structured 0.000000 / 0.000000, combined 0.000000 / 0.000000. Budget usage events by feature: none.

### gamma (total 0.000000 USD)

| Stage | Runs | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|
| brief | 2 | 0 | 0 | 0.000000 |
| embed | 7 | 0 | 0 | 0.000000 |
| extract | 7 | 0 | 0 | 0.000000 |
| ingest | 7 | 0 | 0 | 0.000000 |
| link | 7 | 0 | 0 | 0.000000 |
| lint | 11 | 0 | 0 | 0.000000 |
| proposals | 7 | 0 | 0 | 0.000000 |
| reconcile | 7 | 0 | 0 | 0.000000 |

| Meeting | Runs | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|
| gamma-01 | 6 | 0 | 0 | 0.000000 |
| gamma-02 | 6 | 0 | 0 | 0.000000 |
| gamma-03 | 6 | 0 | 0 | 0.000000 |
| gamma-04 | 6 | 0 | 0 | 0.000000 |
| gamma-05 | 6 | 0 | 0 | 0.000000 |
| gamma-06 | 6 | 0 | 0 | 0.000000 |
| gamma-07 | 6 | 0 | 0 | 0.000000 |

Per query (USD, total / per query): raw 0.000000 / 0.000000, structured 0.000000 / 0.000000, combined 0.000000 / 0.000000. Budget usage events by feature: none.

## Tests covering the mechanical gates

- Zero unapproved task writes:
  - `tests/context/test_pipeline.py::test_seeding_runs_as_the_owner_even_under_a_bot_credential`: passed
  - `tests/context/test_context_owner.py::test_context_resolve_command_is_human_from_the_browser_and_agent_from_dani`: passed
  - `tests/context/test_service.py::test_the_owner_accepts_dismisses_or_defers_agent_suggestions`: passed
  - `tests/context/test_publication.py::test_agents_cannot_approve`: passed
  - `tests/context/test_review_api.py::test_approve_publishes_once_and_agents_are_refused`: passed
  - `tests/context/test_publication.py::test_changed_payload_needs_a_new_approval_and_approved_work_is_never_modified`: passed
  - `tests/context/test_review_api.py::test_updates_need_their_own_approval`: passed
  - `tests/context/test_publication.py::test_link_approval_records_the_task_without_writing_it`: passed
  - `tests/context/test_review_api.py::test_retry_route_is_human_only`: passed
- Zero duplicate task writes:
  - `tests/context/test_pipeline.py::test_replaying_every_job_and_redelivering_change_nothing`: passed
  - `tests/context/test_end_to_end.py::test_fixture_history_converges_across_orders_and_repeats_are_no_ops`: passed
  - `tests/context/test_lint.py::test_ten_runs_without_new_evidence_create_nothing_and_skip_the_model`: passed
  - `tests/context/test_publication.py::test_approval_publishes_exactly_one_task_with_meeting_provenance`: passed
  - `tests/context/test_publication.py::test_crash_during_publish_and_duplicate_delivery_write_one_task`: passed
  - `tests/context/test_publication.py::test_lost_commit_replays_stored_command_results`: passed
  - `tests/context/test_publication.py::test_linear_publication_reuses_the_mapping_so_sync_never_duplicates`: passed
  - `tests/context/test_publication.py::test_remote_timeout_reconciles_under_the_same_issue_before_retrying`: passed
  - `tests/context/test_publication.py::test_unconfirmed_linear_publication_retries_with_the_original_identity`: passed
- Zero cross-owner disclosures:
  - `tests/context/test_service.py::test_hits_for_other_owners_never_leak`: passed
  - `tests/context/test_retrieve.py::test_scope_holds_in_both_legs_and_never_leaks_across_owners`: passed
  - `tests/context/test_service.py::test_a_key_revoked_during_retrieval_gets_nothing`: passed
  - `tests/context/test_sources.py::test_connection_must_belong_to_the_owner`: passed
  - `tests/context/test_context_owner.py::test_session_delete_flag_and_shared_workspace`: passed
- Complete processing accounting for ready sources:
  - `tests/context/test_end_to_end.py::test_fixture_history_converges_across_orders_and_repeats_are_no_ops`: passed
  - `tests/context/test_extract.py::test_long_meeting_full_coverage_and_cost_per_meeting`: passed
  - `tests/context/test_models.py::test_runs_record_epoch_and_watermark_for_staleness`: passed
- Valid surviving provenance on displayed claims:
  - `tests/context/test_apply.py::test_inexact_evidence_is_refused`: passed
  - `tests/context/test_evidence.py::test_deleting_one_of_two_sources_keeps_the_assertion_and_the_last_retracts_it`: passed
  - `tests/context/test_service.py::test_evidence_from_a_deleted_source_is_withheld`: passed
  - `tests/context/test_service.py::test_brief_keeps_only_claims_whose_citations_survive_and_needs_tasks_read_for_task_claims`: passed
  - `tests/context/test_briefs.py::test_source_deletion_stales_and_rebuild_forgets_the_deleted_statement`: passed
  - `tests/context/test_retrieve.py::test_raw_segments_are_annotated_and_stale_passages_stay_out_of_current_answers`: passed
- Concurrency, deletion, pause, undo and replay tests pass:
  - `tests/context/test_pipeline.py::test_a_new_revision_during_the_model_call_makes_the_result_stale`: passed
  - `tests/context/test_pipeline.py::test_pause_rejects_outstanding_results_and_resume_continues_without_new_calls`: passed
  - `tests/context/test_pipeline.py::test_deletion_job_purges_and_retracts`: passed
  - `tests/context/test_pipeline_wiring.py::test_admin_pause_rejects_results_and_schedule_due_requeues_what_is_still_current`: passed
  - `tests/context/test_admin.py::test_pause_rejects_results_of_runs_started_before_it`: passed
  - `tests/context/test_decisions.py::test_wrong_retracts_records_a_negative_and_stales_briefs_and_undo_restores`: passed
  - `tests/context/test_decisions.py::test_undo_refuses_when_a_later_decision_depends_on_it`: passed
  - `tests/context/test_decisions.py::test_decision_seq_is_allocated_per_owner_under_lock`: passed
  - `tests/context/test_sources.py::test_deletion_excludes_now_and_purge_blanks_content_but_keeps_a_tombstone`: passed
  - `tests/context/test_lint.py::test_a_pass_stops_when_the_owner_is_paused_or_the_epoch_moves`: passed
  - `tests/context/test_publication.py::test_paused_processing_still_publishes_approved_work`: passed
  - `tests/context/test_publication.py::test_update_against_a_task_edited_since_is_a_conflict_not_an_overwrite`: passed
  - `tests/context/test_review_api.py::test_stale_revision_refuses_the_whole_batch`: passed
  - `tests/context/test_pipeline_wiring.py::test_a_statement_said_again_after_a_revision_dropped_it_comes_back_unless_marked_wrong`: passed

## Per-history runs

- alpha: labels 42 (0 verified); owner `eval-alpha-arriv-29910c50`; maintenance created {'questions': 0, 'proposals': 0, 'briefs': 0, 'brief_versions': 0, 'tasks': 0, 'assertion_versions': 0}.
- beta: labels 37 (0 verified); owner `eval-beta-arriv-bd0dd1da`; maintenance created {'questions': 0, 'proposals': 0, 'briefs': 0, 'brief_versions': 0, 'tasks': 0, 'assertion_versions': 0}.
- gamma: labels 37 (0 verified); owner `eval-gamma-arriv-95144093`; maintenance created {'questions': 0, 'proposals': 0, 'briefs': 0, 'brief_versions': 0, 'tasks': 0, 'assertion_versions': 0}.
