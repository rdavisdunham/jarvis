# Meeting context pilot: switching it on for one owner

Meeting context ships with `context_enabled = false`. Production is unaffected until you follow these steps.
They switch it on for the server and use it with one owner (you), in your personal workspace only. Background:
[CONTEXT_PLATFORM.md](CONTEXT_PLATFORM.md). Agent contract: [EXTERNAL_AGENTS.md](EXTERNAL_AGENTS.md#meeting-context).

The flag is server-wide, not per owner. While it is on, any owner can submit meetings in their personal workspace;
for the pilot only you hold `sources:write` keys and only you import files.

## 0. Back up first (required)

Railway's point-in-time recovery is **not available** on the pgvector PostgreSQL image production has run since
2026-11-04 (`CLOUD_MIGRATION.md`, "2026-11-04 pgvector image swap"). The most recent verified restore point is the
export taken before that swap. Before enabling the pilot, take a fresh encrypted export and keep it off Railway:

```sh
# From a shell that can reach the production database (as on 2026-11-04: Railway SSH into the Postgres
# service, no public endpoint). JARVIS_BACKUP_KEY is the escrowed key; never print it.
JARVIS_DATABASE_URL=... JARVIS_BACKUP_KEY=... JARVIS_BACKUP_DIRECTORY=/tmp/eridani-backup \
  python scripts/backup.py once
```

Copy the `jarvis-<timestamp>.pgdump.enc` file and its `.json` manifest out, and test-restore it into a
disposable database (`python scripts/backup.py restore --file ... --target ...`) before relying on it. Repeat
before any later change to the pilot (prompt, model, settings) and keep at least the last two exports.

After the pilot has data, also take a context history export (section 7) at the same times.

## 1. Server flag and settings

Set on both the API and worker services, then redeploy both:

| Variable | Pilot value | Why |
|---|---|---|
| `JARVIS_CONTEXT_ENABLED` | `true` | Turns on routes, MCP tools, the worker tick and the Meetings view |
| `JARVIS_CONTEXT_SOURCE_BUDGET_USD` | `1.0` (default) | Per-meeting extraction cap; a meeting that hits it waits for you (`awaiting_input`) |
| `JARVIS_CONTEXT_QUEUE_CONCURRENCY` | `2` (default) | Context jobs in parallel |
| `JARVIS_CONTEXT_MODEL_TIMEOUT_S` | `90` (default) | Per model call |
| `JARVIS_CONTEXT_GRANOLA_ENABLED` | leave `false` unless you have a Granola Business key | Optional poller |

Leave `JARVIS_CONTEXT_MAX_SOURCE_CHARS` and `JARVIS_CONTEXT_POLL_MINUTES` at their defaults. Model spend is
recorded under the owner budget with the cost features `context_extraction`, `context_linking`,
`context_reconciliation`, `context_embeddings` and `context_synthesis` (Settings → System → usage); make sure the
monthly owner budget has room.

Check: signed in, Settings → Connections shows "Meeting sources", and the Meetings view appears in navigation.
`GET /api/v1/context/admin/status` (signed in) returns your `owner_id`, `epoch` 0 and `paused: false`; note the
`owner_id` for the CLI steps below.

## 2. Bot keys and scopes

Create keys in your **personal** workspace (Settings → Connections → agent access). Context scopes are refused in
shared workspaces.

| Key | Scopes | Used by |
|---|---|---|
| Transcript submitter | `sources:write` only | `scripts/bridge_submit.py`, the recorder, a Zoom export script. It cannot read context back. |
| Agent reader | `context:read` (+ `tasks:read` if agents should see linked task state) | Claude Code or another MCP client asking about meetings |
| Agent with suggestions (optional) | `context:read`, `context:write` | Same, plus correction suggestions. Only you can delete a meeting. Suggestions only reach your Questions. |

No key can approve proposals, publish tasks or merge people; those are yours in the web app.

## 3. Bring in meetings

Pick whichever source you have. All of them end in the same ingestion path.

**File import (no account needed).** Settings → Connections → Meeting sources → Import: TXT, Markdown, WebVTT,
SRT or versioned JSON. Speaker labels come from `Name: text` lines or VTT voice tags.

**Folder bridge.** From any machine with the exports:

```sh
ERIDANI_SOURCES_KEY=<sources:write key> \
  python scripts/bridge_submit.py ./exports --base https://<your Eridani origin> --connector recorder --dry-run
# then again without --dry-run
```

It reads `.vtt`, `.srt`, `.md`, `.txt`, envelope JSON and Granola MCP exports (`--connector granola`). A sidecar
`<name>.meta.json` gives title, start time and timezone. Resubmitting an unchanged folder creates nothing; a
changed file creates a new revision. Only import your own meetings, starting with the last 90 days.

**Any client.** `PUT /api/v1/external/sources/{connector}/{external_id}` with the submitter key (see
EXTERNAL_AGENTS.md, "Submit transcripts").

**Granola REST (optional).** Only with a Business or Enterprise workspace API key: set
`JARVIS_CONTEXT_GRANOLA_ENABLED=true`, redeploy, then Settings → Connections → Meeting sources: enter the account email and key, Connect Granola (stored
encrypted, never shown again), Test key, then Sync now. The poller runs every 5 minutes and a 90-day backfill runs
behind new meetings with one consolidated notice. Only notes you own are imported. Without such a key, use the
bridge with Granola MCP exports instead.

## 4. Optional: model-written briefs and other meeting memory settings

Briefs are deterministic by default (restated, cited assertions; no model). Model-written briefs cost one
synthesis call per changed person or project in each meeting and are opt-in per owner:

1. Signed in, in your personal workspace, open Settings → Connections and scroll to **Meeting memory**.
2. Turn on **Model-written briefs**. It saves at once; turn it off the same way.

The same panel holds the other owner settings:

- **Apply reworded replacements** (`semantic_supersede`): leave off for the pilot, so free-text supersessions stay
  in shadow and the current value is unchanged.
- **Weekly cleanup budget** (`lint_budget_usd`): the weekly lint's model cap, default $0.50, from $0 (no model
  checks) to $5.00; each check reserves $0.05.
- **Approved tasks go to** (`review_destination`): the default destination on the meeting review, Eridani or
  Linear. Choosing a destination when approving a review also updates it.

The panel saves through `PATCH /api/v1/context/settings/preferences` (signed-in session with the CSRF header; bot
keys cannot). The deterministic test engines are not a setting and are never used for the live pilot.

## 5. What to expect on the first day

- A submitted meeting is `queued` at once, then `processing`, then `ready`, usually within minutes once the
  worker picks it up (target: 95% of one-hour meetings within 10 minutes). Receipt never waits for a model.
- One "meeting ready" notification per meeting, opening its review in Meetings. A historical import (90-day
  backfill, bridge of old files) produces one consolidated notice per day instead.
- The review shows the brief, extracted decisions, commitments, risks and questions with exact quotes and speaker
  badges, and proposed tasks. Your own clear actions are pre-checked; other people's work and suggestions are
  unchecked. Approving creates exactly one task per proposal (Eridani by default; Linear when chosen). Updates to
  existing tasks need their own approval.
- Uncertain people or links and contested statements appear in Questions. Names alone never create people.
- At 17:00 local time, one digest if anything is still unreviewed, set to Later or deferred; nothing on empty days.
- Agents with a `context:read` key can call `context_search`, `context_brief` and the other tools in
  EXTERNAL_AGENTS.md; every statement cites its meeting.
- A meeting that hits the per-meeting budget shows as waiting for you instead of ready.
- Sunday 03:00 local, the weekly lint runs if memory learning and the dream sequence are on for you.

Check `GET /api/v1/context/admin/status` for failed runs and jobs after the first meetings.

## 6. Pause and roll back

**Pause processing** (stops all automatic extraction and writes; reads, corrections and approved publications
continue; works even with the flag off):

```sh
# Signed-in session: send the session cookie and the X-CSRF-Token header, as the web app does.
curl -X POST https://<origin>/api/v1/context/admin/pause -H "X-CSRF-Token: <csrf>" -b "<session cookie>"
```

`POST /api/v1/context/admin/resume` resumes; work interrupted by the pause is requeued once. Settings →
Connections → Meeting memory → Processing has the same Pause and Resume control, with a status line (paused since,
epoch, runs outstanding); it shows while meeting context is on. With the flag off, use the routes above.

**Rollback drill.** `POST /api/v1/context/admin/rollback-drill` pauses, exports your history and proves it
imports. Its response contains the whole history: save it as you would an export, or discard it.

**Full rollback.** Pause, take a history export (section 7), then set `JARVIS_CONTEXT_ENABLED=false` and redeploy.
Routes and tools answer `CONTEXT_DISABLED`, the Meetings view disappears and the worker tick stops. Tasks already
published stay; context tables and data stay (they are never dropped). Re-enabling picks up where it stopped.
Restoring the database from the section 0 export is the last resort and loses everything since.

## 7. Export your meeting history

```sh
/app/.venv/bin/python scripts/context_export.py --owner <owner_id> --out /tmp/context-history.json
# check a file imports without changing anything:
/app/.venv/bin/python scripts/context_export.py --owner <scratch owner> --in /tmp/context-history.json --dry-run --new-ids
```

The file holds transcripts and context; treat it as sensitive and encrypt it at rest. It leaves out connector
credentials. The account export (Settings → System → data export) contains the same history.
