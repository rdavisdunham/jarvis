"""Versioned tool guidance, pinned when durable work is accepted.

Baseline remains executable for accepted jobs and controlled comparisons. These
policies change exposure/guidance only; authorization, revisions and receipts are
still enforced by the ordinary tool executor.
"""
import copy

POLICIES = ("baseline", "discovery-v1", "reads-v1")
DISCOVERY = (
    "Call an exposed tool directly; do not load its group first. "
    "Use tools_load only for capabilities absent from the current tool list. "
    "Discovery never reads or changes personal records."
)
FRESH_READS = (
    "For a sparse edit, one fresh task_resolve/task_list/task_get or record lookup "
    "from this invocation is enough when it supplies the exact unambiguous ID, "
    "current revision, source status and all fields the change depends on. "
    "Do not add a detail read just because you are editing. "
    "A title match alone is insufficient. Browser context, prior conversations, "
    "saved selections and restored checkpoints are not fresh reads. "
    "Fetch full details for body edits, replacement relationships, truncated content "
    "or missing timing/source fields. Keep omitted fields unchanged. "
    "On a revision conflict, reread and reconsider the requested patch; never just "
    "substitute a newer revision. A committed local mutation receipt is sufficient "
    "to acknowledge that local change without a verification read. A queued external "
    "write is still pending, and browser controls still need displayed acknowledgement."
)


def validate(policy):
    if policy not in POLICIES:
        raise ValueError("Unknown tool policy")
    return policy


def definitions_for(definitions, policy):
    validate(policy)
    if policy == "baseline":
        return definitions
    result = copy.deepcopy(definitions)
    for tool in result:
        name = tool["name"]
        if name == "tools_load":
            # Retain the existing (possibly bot-scoped) group description/schema.
            description = tool["description"]
            groups = description[description.index("Groups:"):] if "Groups:" in description else ""
            tool["description"] = DISCOVERY + " Loaded definitions appear on the next request. " + groups
        if policy == "reads-v1":
            if name in {"task_update", "task_batch", "task_selection_update"}:
                tool["description"] = tool["description"].replace(
                    "Read the current ID/revision first.",
                    "Use a fresh unambiguous lookup's ID/revision; a compact resolver/list result "
                    "suffices for a sparse edit when it contains every needed field. "
                    "Read full details for notes, replacement links or omitted dependencies."
                )
            elif name in {"task_resolve", "task_list"}:
                tool["description"] += (
                    " A new lookup supplies current IDs/revisions, timing and source status. "
                    "For a sparse date/status/priority edit, use this result directly when sufficient; "
                    "task_get is only needed for missing details. Saved selection pages are frozen snapshots. "
                    "notes_preview is not the full body."
                )
            elif name == "record_update":
                tool["description"] += (
                    " One fresh record lookup with schema_revision and expected_revision is enough "
                    "for a sparse edit; fetch record_get only for missing dependencies or full content. "
                    "Do not infer revisions from task IDs or search excerpts."
                )
    return result
