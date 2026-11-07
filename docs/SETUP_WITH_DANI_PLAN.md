# Set up with Dani: interview plan

Status: approved by the owner on 2026-11-07. Design report by a Fable agent; the decisions below are the owner's.

## Owner decisions

1. **Voice or typing, chosen up front.** Before the conversation starts, a step asks: "Want to do this by voice? We recommend a conversation for this. If you can't talk right now, feel free to type." It offers two buttons: primary "Use voice mode" and secondary "Type instead".
   - The two are either/or. In voice mode the composer stays disabled, as it does today.
   - In typed mode, a "Switch to voice" button stays visible for the rest of the flow.
   - Ending voice returns to the same conversation in typed mode.
   - Both modes write one transcript.
2. **Re-running** merges into the existing tree. It adds and renames, and archives only the rows the owner ticks.
3. **Core memories** are ordinary `Memory` rows the owner can edit, with a new pin (`core`). Pin and unpin work on any memory, not only ones from setup.
4. **v1 builds only spaces, areas, goals and projects.** If the interview calls for a new record type (for example Client or Course), Dani suggests "refine in Types & fields" instead.
5. **Terra (`gpt-6.1-terra`) runs everything in setup:** typed turns, voice-delegated setup turns and the synthesis. Background memory learning stays on Luna.
6. **Length:** about 12 questions by default, with a 5-question quick version offered when the owner is short on time.
7. **If conversation history is off,** the setup conversation is kept anyway, because the proposal needs it. Dani says so at the start and offers to delete it at the end.
8. **Naming:** the section is called "About you", to match the maintained summary.
9. **"Use voice mode" starts the mic straight away.** The browser permission prompt is the only confirmation.
10. **Later, not in this plan:** a better onboarding that sets up connections and syncs them first (Google Calendar, Linear, and the connection types still to come), then feeds that data into this interview. This waits until there are more connection types and they're stable. Linear project matching may be skipped there, since no projects exist yet on a first run.

## Current state (2026-11-07)

**Entry points**
- Today: `App.tsx` (around line 2655) renders `<Onboarding compact established=... onTalk={()=>startGuidedConversation(SETUP_REQUEST)}/>` in the personal workspace only. It hides once a space and a project exist (`setup-model.ts:11-12`), or when the status is completed or skipped (`Onboarding.tsx:81-83`).
- Settings → Organization renders the full panel (around `App.tsx:2976`).
- `StructureEditor` "Refine with Dani" uses `typesFieldsRequest`. Keep it.

**Panel** (`Onboarding.tsx:135-182`)
- Heading "Make Eridani yours".
- Buttons: "Set up with Dani", "Use setup form" and "Skip for now".
- Form path: preferred name, time zone, "What do you want to track?", grouping, notes. Then `onboarding.preview` → `SetupProposalDetails` → `structure.apply` + `onboarding.save(status=completed)`.
- The backend (`onboarding.py`) keeps state in `OwnerSettings.values["onboarding"]` and refuses shared workspaces.

**Dani path today**
- `startGuidedConversation` (`App.tsx` around 1782) sends the fixed `SETUP_REQUEST` (`setup-model.ts:14`) as an ordinary durable chat request.
- The agent loads the `setup` tool group (`tool_catalog.py:8`): `onboarding_state`, `onboarding_save`, `onboarding_preview`, `structure_schema`, `structure_apply`, `ui_records`, `routing_state` and `routing_understand`.
- It asks one question per turn with `work_needs_input`; answers continue through `work_answer` and `work_continuation.adopt`.
- In voice, every answer is delegated to the backend, which relays its questions back as commentary.
- The composer is disabled while voice is active, and `sendChat` refuses when `voice.current` is set.

**The gap:** everything today designs the schema (types, fields and homes). Nothing proposes a tree of actual spaces, areas, goals and projects. Reusable pieces:
- the core record tools `space_create`, `area_create`, `goal_create` and `project_create`;
- the preview-hash-then-apply pattern of `record_templates.plan/apply`;
- `OrganizationTree.tsx`, which renders real records, not drafts.

**Memory:**
- `Memory` (`models.py:328-345`) holds content, attribution, tags, evidence, fact_key, fingerprint, embedding, revision, suppressed, supersedes_id and merged_into_id.
- Luna learns facts from chats (`memory_learning.py`).
- `memory.capture` is the manual capture path.
- Shared workspaces block the `memory_*` and `onboarding_*` tools (`access.py:211`).

**In flight (check whether merged):**
- `feat/about-you`: the `about_you_summaries` table, a daily Luna summary, and its injection at voice session start.
- `feat/typed-chat-terra`: the `terra-6.1` profile in `agent_models.catalog()`, a `typed_chat(prefs)` / `chat_profile` preference, `enqueue(..., typed=True)`, and `adopt` carrying the profile.
- `model_map.PROFILES` (background and extraction jobs) has no gpt-6.1 entry yet.

## Experience

Copy is in sentence case, per `docs/DESIGN.md`.

**Launcher** (Today banner and Settings → Organization)
- Panel "Make Eridani yours": "Dani asks about your life and work, then proposes how to organize it and what to remember about you. You review everything before it changes."
- Buttons: primary "Set up with Dani", then "Fill out the form" (today's "Use setup form", with unchanged behaviour), then "Skip for now".
- After the first completion the banner retires. Settings → Organization shows "Update with Dani" and "Fill out the form".

**Step 0: mode choice** (decision 1). It opens the companion in setup mode on a dedicated setup conversation.

**Step 1: the conversation.** Dani's first message is fixed, needs no model call, and is stored as an assistant Source so both models see it:

> Tell me about you. What you do, how your weeks go, what you're juggling, and what you'd like help keeping track of. I'll ask a few questions, then propose how to organize Eridani and what to remember about you.

- Composer placeholder in typed mode: "Type your answer…".
- A progress chip row: "Getting to know you", "Your week", "Areas and goals", "What's in flight", "How to help". The synthesizer's notes tick the chips, not the live model.
- A "Propose my organization" button is always available.

**Question themes.** One question at a time. Never ask what memory or the profile already says; confirm it instead ("I have you down as working at ABC. Still right?").
1. You: name, location, work and/or school, and role.
2. A week in your life: weekdays, weekends, standing commitments, and the people you work with or look after.
3. Responsibilities, which become areas.
4. What you're working toward over the next few months, which become goals.
5. What's in flight now, which become projects.
6. What to track and how Dani should help.
7. Preferences and limits, which become core memories.

Move on after two or three concrete details, or after two short answers. Aim for 10 to 14 questions, with a quick version of 5. After themes 1 to 5, ask: "That's enough for a first pass. Want to see what I'd set up?"

**Step 2: proposed organization.** The `SetupTreeReview` panel opens in the page, and Dani says: "Here's a first draft of your organization. Change anything, or tell me what's off."
- It's an indented list of spaces → areas → goals → projects. Each row has an inline-editable title, a one-line description, a kind chip, a remove `.btn-icon`, and "Add under this".
- Existing matched records show "Already exists" and are kept. On a re-run, rows show "New", "Rename" or "Archive", with Archive unticked by default.
- Impact line: "4 areas, 6 goals and 9 projects will be created. Nothing is created until you confirm."
- Footer: "Keep talking", "Fill in by hand", and primary "Create this organization".
- Reuse `StructureEditor`'s review-footer pattern and `PlacementImpact` styling, not the editor itself.

**Step 3: About you**
- Intro: "These are the things Dani will keep in mind. Edit or skip any of them."
- Candidate rows: an editable sentence, a source chip ("From this conversation" with the quote on hover, or "Already known"), and a check ring that's on by default.
- Note below the list: "Also learned: n facts from earlier chats will stay as they are."
- Footer: "Keep talking" and primary "Save about you".

**Finishing:** "You're set. I'll keep learning as we talk. Update this any time from Settings → Organization or Memory → About you." Then the status becomes completed, the About-you rebuild is queued, and the banner retires. If the owner stops midway, the panel offers "Resume with Dani".

**Re-run**
- Entry points: Settings → Organization "Update with Dani" (focus: organization) and Memory → About you "Update with Dani" (focus: who you are).
- Opening line: "Tell me what's changed since we last talked."
- The synthesis proposes a diff. For the tree: add, rename or archive. For memories: keep, update (`supersedes_id`) or retire (clear `core`).

## Architecture

**State**
- The transcript is the setup `Conversation` and its `Source` rows; there's no parallel transcript.
- New table `setup_sessions`, one row per owner and rewritten per run. Columns:
  - `owner_id` (PK), `conversation_id`, `mode` (voice or typed);
  - `status` (interviewing, proposing, reviewing, completed or abandoned) and `focus` (both, organization or about_you);
  - `notes` (JSONB), `tree_proposal_id`, `memory_candidates` (JSONB) and `confirmed_memory_ids` (JSONB);
  - `started_at`, `finished_at`, `model`, `cost` and `revision`.
- Keep `OwnerSettings.values["onboarding"]` for the form path. `onboarding.save(completed)` also accepts an applied tree proposal.
- New table `organization_proposals`, which mirrors `StructureProposal`. Columns: `id`, `owner_id`, `account_id`, `conversation_id`, `tree` (JSONB), `impact`, `preview_hash`, `applied_at`, `expires_at` and `command_id`.
  - Node shape: `{id, kind: space|area|goal|project, title, description, existing_id?, action: create|keep|rename|archive, children}`.
  - At most 80 nodes.

**Who does what**
- **GPT-Live**, through `live_instructions(prefs, voice, mode="setup")`. It holds the interview itself, with the themes, the move-on rules and the known facts as DATA. Its delegation rule for the session is replaced: it delegates only
  - (a) "propose my organization/about you",
  - (b) a yes to create or save,
  - (c) a revision request, and
  - (d) ending voice.

  Everything else is conversation. The server detects the setup conversation (`conversation_id == setup_sessions.conversation_id`). Raise the connect history window for setup conversations, or pass the synthesizer's notes as DATA.
- **Terra (`terra-6.1`)** handles:
  - the typed interview, through the typed-chat profile with a `purpose: "setup"` field on the chat input and a `SETUP_INTERVIEW` block in `backend_instructions`. This block goes in the dynamic half, to keep the static prefix cache-stable. Setup questions are plain replies, not `work_needs_input`, so voice can take over.
  - voice-delegated setup turns, through an `enqueue` profile override that `adopt` carries.
  - the synthesis job `setup_synthesis`: strict JSON `SetupProposal {notes, tree, core_memories[{content, fact_key, evidence_source_id, evidence, existing_memory_id?, action}]}`, with a new `model_map.Profile("terra-6.1")`.
- **Luna** handles `extract_memory`, embeddings and the About-you regeneration, as today.

**Tree propose and apply (never a silent write)**
- `setup_propose` is a tool in the `setup` group, and `POST /api/v1/setup/propose` backs the button. Both enqueue the synthesis with:
  - the conversation's Sources (the newest 60);
  - `structure.schema_data`;
  - the current spaces, areas, goals and projects;
  - existing memories (a bounded newest-60 read, clipped to 300 characters each);
  - the About-you summary;
  - the previous proposal, on a re-run.
- Validation: every `existing_id` belongs to the owner, kinds exist, there are no duplicate titles under one parent, and depth is at most 4.
- The result is stored with a `preview_hash`, and a UI action opens the review. Extend the `ui_records` proposal action (around `App.tsx:1222`) with `organization_proposal_id`.
- `setup.tree_apply {proposal_id, preview_hash, expected_revision}` runs as one command in one transaction, through the existing create mutations. Revert works through the existing paths.
- Edits in the panel re-preview before apply, and a stale proposal needs a new preview.
- Watch the unique constraints (`uq_area_space_name`, `uq_project_owner_name`): match by name before creating.

**Core memories → About you**
- New indexed `Memory.core` boolean, with `memory.pin` and `memory.unpin` commands.
- Confirming a candidate creates or flags the memory, with attribution via `memory_review.attribution` (`owner_statement` for a human confirmation).
- Dedupe with `fingerprint` and `duplicate_of` against facts the Luna extractor may already have saved from the same turns.
- Unticked candidates are discarded, not rejected.
- After saving, run `about_you.schedule(..., rebuild=True)`. The generator always includes core facts, and the digest includes core ids.
- The Memory page gains an "About you" panel: the summary, a core memory list (pin or unpin on any row) and "Update with Dani".

**Privacy**
- Personal workspace only. The `/setup/*` routes and tools refuse shared workspaces, and `access.py` blocks `setup_*` there.
- Never log transcript text.

**Cost:** about $0.20–0.40 per typed run, with voice adding 8–10 Live minutes. Add a new cost feature, `setup` ("Set up with Dani").

## Build plan (one PR each)

1. **Setup session backend.**
   - `setup.py` (start, state, propose, finish), the `setup_sessions` migration, and the routes.
   - The fixed opening Source, `purpose` on the chat input, and `SETUP_INTERVIEW` in the dynamic half.
   - The alternative completion gate in `onboarding.save`.
   - Tests: personal-only, resume, history-off handling, and the prompt block appearing only for setup.
2. **Companion setup mode (web).**
   - `Onboarding.tsx` becomes the launcher, with the mode-choice step (decision 1).
   - `SetupConversation.tsx`: the opening message, the chips, "Propose my organization" and "Switch to voice".
   - `startGuidedConversation` opens the setup conversation. Keep `typesFieldsRequest`.
   - Tests: browser fixtures select by accessible name, so keep "Set up with Dani".
3. **Voice setup mode.**
   - `live_instructions(mode="setup")`, setup conversation detection, the replaced delegation list, the larger history window, and the profile override for voice-delegated turns.
   - Tests in `tests/test_live_voice.py`.
4. **Synthesis on Terra.**
   - The `model_map` profile and job, the strict schema, validation, the `organization_proposals` table, and the `setup_propose` / `setup_revise` tools.
   - Tests with a stubbed provider, plus evals in `evals/app/features/setup_interview.json`, registered in `features.json`.
5. **Tree review and apply.**
   - `SetupTreeReview.tsx`, `setup.tree_apply`, the UI action, and revert coverage.
   - Tests: hash mismatch, stale schema, revert, and the component.
6. **Core memories and About you.**
   - The `Memory.core` migration, pin and unpin, `SetupMemoryReview.tsx`, the Memory page "About you" panel, and the rebuild after save.
   - Tests: dedupe, attribution and shared-workspace refusal. Depends on `feat/about-you`.
7. **Re-run, finishing and docs.**
   - Focus narrowing, diff semantics, the banner and the entry points.
   - Update `docs/APP_FUNCTIONALITY.md` and add a `docs/TODO.md` entry for the later connections-first onboarding (decision 10).
   - Add the physical phone and voice check to the acceptance list.

## Build log and deviations

- **PR 1** (setup session backend): `apps/api/jarvis/setup.py`, `setup_models.py`, migration `0056_setup_sessions`.
  - The `organization_proposals` table is created here, not in PR 4, so the tree review and apply (PR 5) can ship before the Terra synthesis (PR 4) and `onboarding.save(completed)` can already accept an applied tree.
  - Routes: `GET /api/v1/setup`, `POST /api/v1/setup/start` (`mode`, `focus`, `restart`), `POST /api/v1/setup/mode` and `POST /api/v1/setup/finish` (`status`, `delete_conversation`). `propose` arrives with the synthesis in PR 4.
  - Resuming on another device moves the setup conversation to that device, since it is the owner's own transcript.
  - The interview block is skipped for voice-delegated turns until PR 3 gives them their own block.
  - The `/api/v1/setup` routes are counted under the existing `onboarding` eval feature in the surface inventory.
- **PR 2** (companion setup mode): `Onboarding.tsx` launcher with the mode choice, `SetupConversation.tsx`, `startSetup` in `App.tsx`.
  - The launcher's primary label follows the saved state: "Set up with Dani", "Resume with Dani" (an active run) or "Update with Dani" (after completion). Resuming a typed run skips the mode question; resuming a voice run asks again, so the mic never starts by surprise.
  - The form's "Talk to Dani" button opens the same mode choice, after saving the form's answers as before.
  - Starting Dani from a completed setup no longer resets the onboarding status to in progress.
  - In the setup conversation, the composer's "Talk to Dani" voice button behaves like "Switch to voice".
  - "Propose my organization" appears only when the setup state reports `can_propose` (PR 4).
  - `batch-c.mjs` now walks the new launcher: mode choice, the fixed opening, a typed answer sent with `purpose: "setup"`, and resume.
- **PR 3** (voice setup mode): `LIVE_SETUP` in `agent_instructions.py`, `live_instructions(..., mode="setup", setup_notes=...)`.
  - `LiveController.connect` detects the setup conversation (`setup.active`), swaps in the setup delegation policy, passes the synthesizer's notes as DATA, and reads the newest 40 transcript sources (`LIVE_HISTORY_WINDOW`) even with history off, since the setup transcript is always kept.
  - Voice-delegated setup turns get a narrower dynamic block (`SETUP_VOICE_TURN`) instead of the interview block.
  - `enqueue` runs every turn in an active setup conversation (typed or voice-delegated) on `terra-6.1` when an OpenAI key is configured, whatever the typed-chat or task-agent choice; continuations keep it through `adopt` (#191). Bot requests are never overridden.
- **PR 4** (synthesis on Terra): `apps/api/jarvis/setup_synthesis.py`, model map profile `terra-6.1` and job `setup_synthesis` (cost feature `setup`, "Set up with Dani").
  - `setup_synthesis` is pinned to Terra in `model_map.PINNED`; every other structured job keeps Luna. The drift guards in `tests/test_model_eval.py` accept pinned jobs explicitly, and `scripts/model_eval.py` has GPT-6.1 Terra's price.
  - `POST /api/v1/setup/propose` (the button) and the `setup.propose` command (tool `setup_propose`) queue one worker job; a second press while one is pending returns the same job. There is no separate `setup_revise`: the synthesis reads the conversation, so a change the owner just asked for is in it, and it gets the previous proposal to edit.
  - The model sees short refs (`s1`, `r1`, `m1`), never IDs. Invented refs are dropped, rows that cannot be placed (wrong parent kind, too deep, past 80 rows) are dropped with their children, and the result goes through `setup_tree.preview`, so nothing is written to records.
  - The tool waits up to 45 seconds for the job so Dani can describe the proposal in the same turn. The button path posts Dani's fixed message ("Here's a first draft…") in the setup conversation instead. A failure always posts a short message and returns the run to interviewing.
  - The review opens on its own when a proposal finishes: the app listens for a new `setup.changed` event (and polls every 5 seconds while proposing) rather than a `ui_records` action, since the proposal arrives after the turn that asked for it.
  - New tool group `setup_interview` (`setup_state`, `setup_propose`, `setup_tree_apply`). `setup_tree_apply` is available to Dani only after the owner's clear yes; the strip and review stay the main path.
  - Core memory candidates are stored on the session for PR 6. Unknown memory or source refs are dropped; an update of an unknown memory becomes an add.
  - The eval feature `setup_interview` (25 scenarios) now owns the `/api/v1/setup` routes and `setup.*` commands in the surface inventory.
- **PR 6** (core memories and About you): migration `0057_memory_core`, `apps/api/jarvis/setup_memory.py`, `SetupMemoryReview.tsx`.
  - `setup.memories_save {choices: [{id, content, include}]}` saves the ticked suggestions as edited, in one command. Add flags a live fact with the same wording instead of saving a copy (fingerprint match against facts the extractor learned from the same turns); keep flags the existing fact, and edited wording supersedes it like Correct; update supersedes; retire clears `core` and keeps the fact. Unticked suggestions are discarded, not rejected. Confirmed facts are `owner_statement` (a human command) and each gets its own explicit Source, so deleting the setup transcript later does not drop them.
  - Dedupe is by fingerprint only, not `duplicate_of` (which needs embeddings the new fact does not have yet). The weekly memory review still merges near-duplicates.
  - `memory.pin` / `memory.unpin` (tools `memory_pin` / `memory_unpin` in the memory group); a correction keeps `core`. The setup_interview group also gets `setup_memories_save`, for a spoken yes.
  - About you has no `rebuild` flag: saving or pinning calls `about_you.schedule`, and `gather` puts pinned facts first under `core_facts`, so the digest changes and the job rebuilds. Owners with no pinned facts keep the same material and digest.
  - The Memory page's About you panel lists pinned facts with Unpin, and every memory row gets Pin to About you. Its "Update with Dani" entry point arrives with the re-run in PR 7.
  - The review opens after the organization is created (or on its own when a proposal has only facts), and from "Review what to remember" in the setup strip.
- **PR 7** (re-run, finishing and docs):
  - Settings → Organization's "Update with Dani" re-runs with focus `organization`; Memory → About you has its own "Update with Dani" (focus `about_you`), which asks the same voice-or-typing question in a small dialog. An active run is resumed as it is, whatever the entry point.
  - Diff semantics needed no new backend: the synthesis already proposes keep, rename, archive and add against the existing tree and memories, and existing records it leaves out stay unchanged. The tree review's intro says so on a re-run.
  - `POST /api/v1/setup/finish` with `completed` posts Dani's closing line, marks the form path's status completed when a tree was applied or facts were saved (so the banner retires), and queues the About you rebuild. Finishing happens after the About you review, after creating an organization with no fact suggestions, or from "Finish setup" in the strip. With history off, a dialog first offers to delete the setup conversation (decision 7).
  - Docs: `docs/APP_FUNCTIONALITY.md` §42, the decision 10 entry in `docs/TODO.md`, and a physical phone and microphone row in `docs/RELEASE_ACCEPTANCE.md`.
- **PR 5** (tree review and apply), built before PR 4: `apps/api/jarvis/setup_tree.py`, `SetupTreeReview.tsx`.
  - Commands `setup.tree_preview {tree}` and `setup.tree_apply {proposal_id, preview_hash}`. Apply has no `expected_revision`: the preview hash covers the tree and the revision, name and archived state of every existing record it names, so any later change makes the preview stale (`PREVIEW_REQUIRED`).
  - Created rows are matched to existing records by name before creating (spaces and projects across the owner, areas within their space, goals within their home), which also avoids the unique constraints. A match with an archived record, duplicate names under one parent, or a project name already in use are blocking issues.
  - A project under a goal is homed where the goal is and linked to it. Archive rows apply only with `archive_confirmed` (the tick), which is off by default.
  - Revert undoes the whole apply as one group (`record.restore_contents` dispatches to `setup_tree.restore`): created rows are archived deepest first, renamed and archived rows are restored. It is refused once anything changed or other records (such as a new task) point into the created rows.
  - The review opens from a "Review proposed organization" button in the setup strip while the session is reviewing. The `ui_records` action for opening it arrives with the synthesis (PR 4).
  - Existing rows keep their descriptions; only created rows take the proposed description.
