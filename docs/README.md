# Eridani documentation

Start with the [active TODO](TODO.md) for what to do next.
The [web v1 completion PRD](ERIDANI_WEB_V1_PRD.md) defines the finish line;
the [response-speed plan](ERI_LATENCY_PLAN.md) defines the audited latency work.

## Current planning and implementation references

| Need | Read |
| --- | --- |
| Priorities, pending acceptance and explicit deferrals | [TODO](TODO.md) |
| Release scope and product decisions | [Web v1 PRD](ERIDANI_WEB_V1_PRD.md) |
| Latency measurements and safe optimization sequence | [Latency plan](ERI_LATENCY_PLAN.md) |
| Batch A implementation and pending real-device checks | [Batch A verification](BATCH_A_VALIDATION.md) |
| Latest audit/repair handoff | [Claude session summary](CLAUDE_SESSION_SUMMARY.md) |
| Feature and evaluation map, dated September 19 | [App functionality](APP_FUNCTIONALITY.md) |
| UI conventions and design tokens | [Orbit design system](DESIGN.md) |
| Flexible records, field learning and notifications | [Configurable planner](CUSTOM_PLANNER_IMPLEMENTATION.md) |
| Durable execution and action cards | [Background work](BACKGROUND_WORK_PRD.md), [clarification continuations](CLARIFICATION_CONTINUATIONS.md) |
| Notes/list organization and search vocabulary | [Note lists](NOTE_LISTS.md), [semantic search](SEMANTIC_SEARCH.md) |
| Account/workspace boundaries | [Accounts and sharing](ACCOUNTS_AND_SHARING.md) |

Feature guides describe the implementation at their stated review dates; source code
and newer accepted decisions take precedence. They are not proof of the current
deployed version or completed device acceptance.

## Operations and integrations

- [Cloud runbook](CLOUD_MIGRATION.md): includes the September cutover record.
  Verify current provider inventory before infrastructure changes.
- [CI and deployment checks](CI.md), [cost tracking](COST_TRACKING.md),
  [prepared R2 backups](R2_BACKUPS.md), [public site deployment](PUBLIC_SITE.md).
- [Google](GOOGLE_SETUP.md), [Linear](LINEAR_SETUP.md),
  [external-agent API/MCP](EXTERNAL_AGENTS.md).
- [Luna](LUNA_SETUP.md) and [optional Gemini](GEMINI_SETUP.md) configuration.
  Gemini comparisons are paused; a Luna no-reasoning profile is planned, not yet shipped.

## Evaluation evidence

Use the [app eval guide](../evals/app/README.md) for today's runner and commands,
and [findings](../evals/app/FINDINGS.md) for repairs versus historical results.
[Langfuse](LANGFUSE.md) explains exporting synthetic runs.

The older [expert protocol](EXPERT_AGENT_EVALUATION.md) and [results](EXPERT_AGENT_RESULTS.md),
[tool-refinement protocol](TOOL_REFINEMENT_EVAL_PLAN.md),
[tool-refinement results](TOOL_REFINEMENT_RESULTS.md),
[held-out protocol](RELIABILITY_HELDOUT_EVAL_PLAN.md) and their related reports are
historical evidence. They stay at their existing paths because fixture exporters
and frozen HTML reports reference them. Their old model choices, counts and costs
are not current defaults. Raw [September artifacts](evals/) and the
[audit source snapshot](AUDIT_SOURCE_SNAPSHOT.json) remain unchanged.

## Archive policy

The [archive index](archive/README.md) contains superseded plans, old release
validation, earlier audits, the complete prior TODO ledger and the retired
prototype README. These are evidence and context, not active operating instructions.
No requirement is considered completed merely because its old document was archived.
