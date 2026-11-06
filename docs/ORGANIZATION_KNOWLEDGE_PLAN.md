# Organization knowledge: living overviews for clients and projects

Status: agreed with the owner on 2026-11-06. Phase 1 built on 2026-11-06 (see "Phase 1 as built").

## Goal

Each specialized bucket, mainly clients and projects, builds up a distilled
understanding of what it is: what the work covers, its vocabulary, its people
and its recent history. Eridani then has an intuition about each bucket. It can
place new work by meaning, and it can answer "what's going on with ABC?" with
real context. This history is what makes Eridani a memory system rather than a
task list.

## Decisions

- **Overviews are capped and distilled**, about one page, and versioned. The full
  history stays attached as linked tasks, meeting facts and people. It is not
  appended to the overview.
- **Meeting briefs stay their own thing.** They remain per-meeting, per-entity
  artifacts, but they feed the bucket overview. Meeting facts add to background,
  people, open questions and recent changes.
- **Linear tasks count.** Tasks that Linear sync places in a bucket feed its
  overview, the same as tasks the owner files. They do not create title-phrase
  rules, because the Linear mapping already places them.
- **The owner's description is the seed.** The existing "Record content" text,
  for example "ABC: home and commercial services, started as pest control, now
  lawn, home and appliance care; we mainly work with their Customer Service
  team", is the fixed starting point. It is never overwritten.
- **Cadence:** update in daily batches, not on every task (default; revisit if
  it feels stale).

## Phases

1. **Living overview per bucket.** A daily job takes the seed description,
   the current overview, and what is new since the last version: tasks with
   their titles and notes (Linear included), meeting facts linked to the bucket,
   and new briefs. It returns a revised overview with these sections: what it
   is, scope and themes, vocabulary, people, recent changes, open questions.
   The record card shows it with its version history. Build on
   `context/briefs.py` (`ContextBrief`, `gather`, `synthesize_request`,
   `mark_stale`) and the deferred TODO "include a project's unlinked planner
   tasks in briefs".
2. **Linear feeds learning.** Add a `linear_mapping` evidence origin
   (`routing.RoutingObservation`, set in `linear_structure.move` / `place`).
   It counts as overview input, not as phrase-rule evidence.
3. **Placement by meaning.** When no phrase rule matches, `routing.suggest`
   scores the new task against each bucket's overview and recent tasks. It uses
   embeddings over `SearchDocument` vectors with a margin, like
   `context/native/link.py`, and optionally a cheap model check. It places the
   task automatically above a confidence bar and asks below it. Example: "QA
   dispute and score-correction process" goes to Transcript Intelligence.
4. **People and Dani.**
   - Store non-exclusive person-to-bucket links with evidence, built from
     meetings and assignments. Today the person page only infers projects
     when it loads.
   - Give Dani a tool to read a bucket's overview, people and meeting facts.
     The `context_*` tools are external-only today.

## Current state (origin/main 19c7afc)

- Rules are whole-word title-phrase matches learned from manual filings only
  (`routing.py`). `MIN_RULE_SUPPORT = 3`.
- Record bodies, task notes and embeddings are not used for placement.
- Four filters keep Linear placements out of learning on purpose.
- Meeting briefs exist per project, client, area and goal, but use only meeting
  facts. They show only on person pages and meeting review.
- Memories are scoped to the owner only. Meeting facts never become memories.

## Phase 1 as built

`apps/api/jarvis/record_overviews.py`, tables `record_overviews` (versions, never edited) and
`record_overview_states` (what the latest version was built from), migration 0052.

- **Buckets:** active `client` and `project` records (`BUCKET_TYPES`). Projects carry their own task row, so the
  record type decides, not `task_id`.
- **Material:** the seed (record content, plus the legacy project description when it adds something), up to
  2,000 tasks in the subtree (child clients and projects excluded; Linear tasks included, marked with their
  identifier), child clients and projects, meeting facts about the record whose evidence survives, and the latest
  non-stale meeting brief (next-meeting prep left out).
- **Cadence:** the worker's minute sweep (`queue_due`) looks at each bucket at most once a day. It compares digests
  per task, child, meeting fact, brief version and seed with `seen`, and queues a `record_overview` job only on a
  change. No change means no model call. A failed or budget-deferred attempt retries after 6 hours.
- **Update vs rebuild:** normally the model gets the previous overview plus only what changed. It rebuilds from all
  the material when there is no version yet, when a meeting fact it used has been withdrawn or deleted (so it
  cannot linger), or when the owner presses Rebuild.
- **Output:** what it is; scope and themes (8); vocabulary (12); people (10); recent changes (6); open questions
  (5). Each item is under 240 characters, enforced again on the way in.
- **Model:** Luna through `extraction_request`, cost feature `record_overviews` ("Client and project overviews
  (Eridani)"), model-map job `record_overview`. One call is a few thousand tokens, well under a cent.
- **UI:** an Overview section on client and project record cards, with version history, Refresh and Rebuild.
  Without a description it explains what to add.
- **Endpoints:** `GET /api/v1/structure/records/{id}/overview[?version=n]` and
  `POST .../overview/refresh {rebuild}`.
- **For later phases:** `record_overviews.for_prompt` returns the latest overview as plain text, for placement by
  meaning (phase 3) and Dani's read tool (phase 4).

