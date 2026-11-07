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

- **PR 1** (setup session backend): `apps/api/jarvis/setup.py`, `setup_models.py`, migration `0055_setup_sessions`.
  - The `organization_proposals` table is created here, not in PR 4, so the tree review and apply (PR 5) can ship before the Terra synthesis (PR 4) and `onboarding.save(completed)` can already accept an applied tree.
  - Routes: `GET /api/v1/setup`, `POST /api/v1/setup/start` (`mode`, `focus`, `restart`), `POST /api/v1/setup/mode` and `POST /api/v1/setup/finish` (`status`, `delete_conversation`). `propose` arrives with the synthesis in PR 4.
  - Resuming on another device moves the setup conversation to that device, since it is the owner's own transcript.
  - The interview block is skipped for voice-delegated turns until PR 3 gives them their own block.
  - The `/api/v1/setup` routes are counted under the existing `onboarding` eval feature in the surface inventory.
