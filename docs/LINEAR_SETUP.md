# Connect Linear to Eridani

In **Settings → Linear**, paste a personal API key, connect, choose teams and save
the sync scope. The default imports only issues assigned to you. Turn that option
off to import all issues in the selected teams.

Create the key in Linear's **Settings → Security & access → Personal API keys**.
Grant read, create and write access, restricted to the teams you want Eri to use.
The app verifies the workspace/member identity and stores the key encrypted with
the existing integration encryption key. It never returns the key to the browser,
puts it into command receipts, or supplies it to the assistant.

## What syncs

Issues become linked tasks. Title, description, status, due date, priority,
assignee and project synchronize. Imported parent relationships link issues that
are both in the selected scope. Linear labels appear in the source metadata;
Eridani tags remain independent. Synced projects are marked “· Linear”.

Edit a linked task normally, or expand **Linear workflow** in task details for
the exact Linear status, priority, assignee and project. **Publish to Linear**
creates a new issue for an existing local task. Eri has the same read/write tools.
Publishing a routine template is disabled; publish a concrete occurrence instead.

Due times/time zones, reminder alerts, work blocks, local tags, work types and
linked notes remain in Eridani. A due date is not a scheduled block.
Local archiving stays local. Completing a linked task also completes its alerts.

Polling runs every five minutes and uses an overlapping updated-time cursor.
A full reconciliation runs daily. Every page must succeed before applying a
snapshot. Removed, inaccessible or out-of-scope remote issues leave their local
tasks available for review. Disconnecting removes the stored key and retains
local records. Unlinking a task keeps it local and stops importing that issue.

## Optional: file Linear teams and projects into Homes

After you choose teams, **Settings → Linear** offers **Review mapping**, or **Skip,
import normally** (the behaviour above). You can open the review again later from the
same place. The mapping is optional; skipping it never blocks the import.

1. Say what a Linear team and a Linear project are in Eridani: an existing type, a
   new type you name, or "not a Home". Eridani never assumes a team or project is a
   Client; preview refuses to continue until you answer for every kind you map.
2. For each team and project choose **Create a new …** (title defaults to the Linear
   name; a project can go inside its team's Home or at the top level), **Use an
   existing Home**, or **Skip**.
3. **Preview** is a dry run of the same code that applies the mapping. It lists the
   structure changes (a new type, a type allowed inside another), the resulting tree
   with the Linear issue count landing in each Home and how many stay unfiled, and
   what happens to issues already imported: filed in their new Home, already there,
   kept where you put them, or following their parent issue.
4. **Apply mapping** applies the previewed structure change through the guarded
   structure preview/apply, creates the Homes and files the imported issues, in one
   transaction. A failure leaves nothing half-applied; apply again. A retried apply
   of the same preview does nothing twice. A preview older than a change to your
   mapping, Homes or structure is refused; preview again.

The mapping is stored on the connection by Linear team/project ID, so later syncs,
repeated setup and a reconnect to the same Linear workspace reuse the same Homes and
types instead of creating new ones. Each newly imported issue is filed in its
project's Home, else its team's Home. Subtasks stay under their parent issue.

Your own filing wins. An issue moved to another Home in Eridani (or filed there
before the mapping) is a local exception: sync and later remaps leave it, and the
remap preview lists it as kept. Remapping a project moves only the issues still in
the Home the mapping gave them.

Filing an issue in a Home never changes its Linear project. A Linear project changes
only when you edit it in the task's Linear section (the write-back above); once Linear
confirms the new project, the issue follows that project's mapped Home. A Home
created from a Linear project follows a rename while its title is still the old
Linear name; a title you changed is kept. Archiving a project in Linear marks it
archived in the review and never archives the Home or its tasks. Connecting a
different Linear workspace starts with no mapping. Skipping after a mapping stops
filing new issues; filed issues stay where they are.

Browser routes (personal workspace, cookie + CSRF), all under `/api/v1/integrations/linear/mapping`:
`GET` (saved mapping, catalog with counts, types and candidate Homes), `POST /discover` (reads teams,
projects and issue counts from Linear; nothing else changes), `POST /preview` (dry run; makes the
structure proposal when types change), `POST /apply` (`fingerprint` from the preview) and `POST /skip`.
`GET /api/v1/integrations/linear` reports `structure.offer` while the choice is still open. Eri has no
tool for this yet; the owner reviews it in Settings.

Implementation: [linear_structure.py](../apps/api/jarvis/linear_structure.py),
[LinearMapping.tsx](../apps/web/src/LinearMapping.tsx); tests:
[test_linear_structure.py](../tests/test_linear_structure.py).

## Write outcomes and conflicts

The app saves a durable job and shows pending until Linear confirms. Creates use
a stable, client-generated UUID, so a lost response can be reconciled without a
duplicate issue. Updates first read the issue and compare its updated time with
the last shared version. An unexpected change opens a difference for review.

In task details, **Review Linear copy** shows both versions. Choose the Linear
version, keep the Eridani version, or unlink. A short-lived comparison token binds
that choice to your task revision, connection and the version you reviewed.
Unknown outcomes stay visible; do not create a second issue to retry one.

Linear's public issue-update mutation does not expose a conditional revision
parameter. The pre-write comparison detects known conflicts, but a remote edit
can still race between that read and the mutation. Writes send only changed
fields to reduce that risk. This is a provider limitation, not an atomic compare
and swap guarantee.

Projects offered for publishing/editing are those discovered on synced issues.
Project creation, labels, comments, cycles, attachments and remote issue deletion
are not write tools in this release. Invitations or Slack messages are not sent.

## API rather than a Slack relay

The built-in Linear Agent is available in Linear and Slack. No documented public
endpoint was found for asking that built-in agent to act from Eridani. Linear's
Agents APIs instead support bringing our own agent into Linear.

Eridani uses the documented GraphQL API for both durable sync and assistant
actions. This avoids maintaining a second, independent MCP write path. Polling
works with the current Tailnet-only deployment; it needs no public webhook
endpoint. OAuth, webhooks and an Eri agent inside Linear can be added later.

Sources:
- [GraphQL API and personal keys](https://linear.app/developers/graphql)
- [Pagination](https://linear.app/developers/pagination) and [filtering](https://linear.app/developers/filtering)
- [Linear Agent](https://linear.app/docs/linear-agent)
- [Building agents](https://linear.app/developers/agents)
