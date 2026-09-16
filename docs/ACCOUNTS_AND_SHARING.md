# Accounts and sharing

Updated September 16, 2026.

## Use it

Open **Settings → Sharing**. Create a shared space or project, invite a person's
Google email as an editor or viewer, and share the app's address. Invitations last
seven days and are bound to the verified email; the app does not send email.

The invited person signs in with Google and accepts the invitation in Sharing.
The workspace selector switches between Personal and shared work. Use
https://app.eridani.app; Tailnet access is no longer required. If Google's OAuth
project remains in Testing, connecting Calendar requires being on its test-user
list. Basic sign-in requests only OpenID/email, for which Google documents an
exception to the test-user restriction; Eridani still requires its own invitation.

The server owner can invite a standalone account without sharing a workspace.
Pairing/PIN sign-in is disabled on the public cloud deployment. Invited people
use Google sign-in and can disconnect Calendar without unlinking their only login.

## Access model

Each account has a private personal record scope. Each shared space/project has
a separate scope and membership list. A project workspace starts with a project
and its space; a space workspace starts with a space. New tasks/notes default to
that root. Members can organize work with the same task, note and calendar tools.

Membership covers the selected shared workspace's tasks, notes, relationships,
local calendar entries and task-alert records:

- **Owner:** edit records and manage invitations, members and roles.
- **Editor:** read and edit workspace records.
- **Viewer:** read records and ask Eri about them; mutations are rejected.

Invitations and role changes require the owner and CSRF validation. Role changes
use revisions. Assignment is a reference to a person/agent, not an access grant.
A member's new tasks default to their own assignee entry. Command receipts record
the acting account separately from the workspace scope.

Shared workspaces start empty. Existing private projects/tasks/notes are not
moved automatically. Private-to-shared graph moves, mixed-workspace aggregate
views and finer per-record permissions are future extensions.

## Privacy and processing

Personal settings, learned memory, Google/Linear credentials and external
calendar data stay in Personal. Shared workspaces hold explicit local appointments
and task work blocks. Publication and private-memory tools require Personal.

Shared conversations are temporary and do not feed personal memory. Eri receives
the workspace, member role and current person's display name, timezone and chosen
backend model. Shared note embeddings belong only to that workspace. Saved task
views are private to each person within each workspace.

Shared alerts appear in that workspace. Personal push subscriptions are not
copied into shared workspaces. Per-member shared push routing remains a future
notification expansion.

## Enforcement and revocation

The existing owner_id column is a record namespace: a personal account or shared
workspace. Existing scoped reads, joins and foreign-ID checks remain intact;
shared data is not unioned with private data.

Membership is checked on authenticated requests, inside command transactions,
before/after tool retrieval, and before returning long HTTP responses. Commands
and membership changes serialize through an access lock. Workspace creation
transforms retry IDs into caller-specific server IDs and cannot claim another
person's private namespace.

Switching rotates device/conversation context. Old tabs reload and conversations
cannot keep acting in a previous workspace. SSE checks access every half second;
revocation clears the shared view and returns the user to Personal. Live checks
access every second and before delegated tools. Already committed workspace
records remain after their author's membership is revoked.

Background indexing and reminders continue in the record's workspace. They never
gain access to a member's private records or credentials. Calendar/Linear polling
and history retention now run for every account.

## Storage and operations

Migration 0012_accounts adds accounts, shared workspaces, memberships and
invitations, plus nullable active-workspace and command-account fields. Existing
records, integrations and sessions retain their identifiers. Legacy owner
metadata is initialized without moving those records.

Encrypted PostgreSQL backups include the new tables. Downgrading after creating
shared accounts/workspaces removes the new membership data; recover from an
encrypted database backup if a rollback must preserve it.
