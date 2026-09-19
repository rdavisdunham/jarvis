# Notes lists and automatic organization

Implemented September 19, 2026.

## Using it

Open Notes and choose **Set up suggested lists** for Movies, Books, Shows,
Restaurants and Recipes, or create a list of your own. Every list has an editable
description and saved filters. Eri can discover the lists, open one, edit its
definition, file a note, or queue organization through the same validated commands
as the interface.

A list is a view of notes, not a folder. Tags are the simplest filter. A content
collection can also supply custom classification fields such as genre, client or
a user-defined viewing status. All selected filters must match. Multiple lists
can display the same note. Removing a list keeps its notes.

New and edited notes are organized in the background when at least one list
enables automatic organization. Existing notes are not silently backfilled: open
one and choose **Organize now**. Automatic filing and individual-item extraction
have separate switches on each list. Generated items do not recursively extract
more items.

For example, “Sam recommended Arrival and Dune—watch these” can create two saved
items. Each links to the unchanged source and retains an exact source passage.
A note already titled Arrival is classified directly. Passing mentions, negative
instructions, hypothetical examples and quoted instructions are excluded.

**Find possible missing items** runs broader meaning search and shows a separate
set of possible matches. Opening or seeing a match does not file it. **File in
list** applies that list's tags and classifications explicitly.

## Storage and execution

Migration 0018 adds three shared tables:
- note_lists: workspace, name, description, filter JSON, switches and revision.
- note_organizations: processing state, source fingerprint, manual-tag lock and
  generated-item marker.
- note_entry_sources: source note, saved note, exact evidence, source revision
  and a unique source/item identity.

Authored content remains in Note and the existing StructureRecord registry.
There are no Movies, Books or Restaurant tables. Source quotes stay in provenance;
they are included in the derived search index without copying them into the
saved item's editable body. The ordinary encrypted database backup includes all
three tables; account export includes their workspace-scoped rows.

A note write queues an organize_note Job and Outbox entry. The existing DBOS
memory/organization queue processes it with bounded concurrency. The worker:
1. Checks ownership, active source and current write permission.
2. Captures the source, list descriptions and bounded existing-entry candidates.
3. Calls Luna through the existing extraction service with strict structured
   output. No database transaction is held across the provider request.
4. Rechecks the note, custom record, list and schema versions and write permission.
5. Validates evidence, confidence, saving intent, compatible filters and reuse
   identity, then saves changes under the workspace mutation lock.
6. Records source links and publishes note/search refresh events.

Provider output proposes changes; it does not define permissions or execute tools.
Evidence must occur exactly in the note; a proposed item's title must occur in
that evidence. New classification needs confidence of at least 0.90; reuse needs
at least 0.95 plus one unambiguous supplied title and unchanged source content.
Matching remains conservative: ambiguous names stay unresolved rather than
merging remakes, editions or places. The candidate context includes the latest
100 other notes; a global title collision check prevents blind duplicate creation
outside that window.

Manual tag changes lock automatic tags, including clearing all tags. Automatic
custom-field assignments only fill missing values. Reprocessing a previously
extracted item preserves subsequent edits and archive decisions. Changed input
invalidates stale model output. Provider failures retry up to five attempts;
the source remains saved and the interface offers a manual retry.

## Learning and permissions

Weekly organization review now also uses independent human note classifications.
It proposes note rules for explicit review, even with strong measured support.
Model-generated organization is excluded from human training examples. A later
manual contradiction pauses the rule. This is organization learning, separate
from personal memory extraction; this worker never writes personal memories.
Automatic note rules only add tags associated with enabled automatic lists.

Lists and source links are account/workspace-scoped. Viewers can read but cannot
file or organize. Shared-workspace jobs recheck the requesting member's role
before applying results. Revoked permission cancels the work. External bot note
writes cannot launch broader automatic writes under the account owner's authority;
notes:read credentials can inspect lists and their contents.

## Boundaries and verification

Custom-field list filters currently compare values stored directly on a content
record, not inherited values. Renaming or removing schema fields can invalidate a
filter; the list then asks to be repaired rather than silently showing everything.
Existing data is not rewritten by migration or suggested-list setup.

Automated coverage includes extraction and classifications, exact-source checks,
duplicates and reuse, manual corrections, stale results, schema changes, shared
permissions, failed-job recovery, custom filters and weekly rule approval.
The browser suite covers source navigation, Eri list navigation, list creation
and editing, Back navigation and layouts at 390, 600, 820 and 1440 pixels.

A bounded real-Luna smoke evaluation passed six synthetic cases: two films,
passing mention, negative instruction, existing single item, mixed task/movie
content, and quoted instruction. This is behavioral evidence, not a comprehensive
quality benchmark. Run scripts/evaluate_note_lists.py --run against a disposable
local database to repeat it.

Release checks: 673 backend tests passed, one skipped; 131 frontend tests passed,
one skipped; the production build, migration/model agreement and all five browser
acceptance suites passed. Notes layouts were visually checked from the generated
mobile screenshot. Physical fold-phone use and spoken GPT-Live interactions remain
on TODO.md.
