"""Persistent task proposals from a meeting's commitments and proposed actions (plan_final.md section 7).

`generate(db, owner, source_id, revision_id)` turns the extract run's `ProposedAction`s
(`stage_results["actions"]`) and the source's live commitment assertions into `context_proposals` rows. It is
deterministic (no model call) and runs as the `context_proposals` job after reconciliation (`perform`).

Identity. `fingerprint` is the proposal's stable identity, independent of evidence: sha256 of the responsible key
(actor ID, else the normalized mention, else "unassigned"), the action key (the title's content words without
stopwords, dates or the responsible person's name) and, for update and link actions the engine tied to existing
work, that work. Regeneration matches drafts to the source's proposals by fingerprint, then by action key alone
(responsibility resolved since), then by a near-equal action key with the same responsible key (a revised quote),
so a proposal keeps its ID across chunk overlap, retries and revisions. Two actions supported by one quote have
different action keys and stay two proposals. Drafts with one identity, or tied to the same task, merge.

Matching existing work (similarity never authorizes mutation):
1. stable: the engine's update or link target, a task or synced Linear issue the commitment is linked to
   (`views.linked_tasks`: subject or object, live entity links, published proposals citing it), or the task
   another meeting's proposal with the same identity published;
2. reversible candidates: an open task whose title shares most content words becomes an unchecked link proposal.

Recommendation (plan section 7 table): a clear unfinished action of the user's own with no match is a checked
`create`, unless the action pass marked it `implied_by_discussion` ("Implied by the discussion", unchecked) or its
checker did not confirm it (unchecked; basis and `verified` come from the extract run's
`diagnostics["actions"]["items"]` by action index); someone else's work an unchecked "Waiting on <name>"
`create`; an owner the action pass inferred but could not prove (`inferred`) an unchecked "Likely yours" /
"Likely <name>" `create`; unclear ownership an unchecked `create` with its reason; covered work a `link`;
changed existing work (due date or assignee differ from the task) an unchecked `update` with `before` and `after`;
an engine `suggestion` stays a separate unchecked suggestion and is never recorded as a promise. Hypothetical,
proposed, negated and question commitments, completed work and retracted interpretations stay context only.

Material changes. A proposal's payload carries `material` (kind, assignee, due date, project, target) and
`wording` (title) hashes. When regeneration changes them, an open proposal gets a new revision with an
`explanation`; a dismissed one stays dismissed unless `material` changed, then it reopens as a new revision with
the explanation; an approved or published one is never modified: its changed work becomes a separate follow-up
`update` proposal against the published task. Open proposals the current, fully extracted revision no longer
supports are superseded. Every revision bump invalidates earlier approvals (`decisions.approve` checks it).
"""

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import exists, select

from ..domain import DomainError, emit, preferences
from ..models import LinearIssue, Task, now
from . import apply, decisions, views
from .models import (
    ContextAssertion,
    ContextEntityLink,
    ContextEvidence,
    ContextOwnerState,
    ContextProposal,
    ContextRun,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from .schema import ProposedAction

OPEN = ("pending", "selected", "later")
LOCKED = ("approved", "published")
DESTINATIONS = ("eridani", "linear")
NEAR = 0.5  # action-key Jaccard for "the same action, reworded" within one source
CANDIDATE = 0.6  # title Jaccard for a reversible existing-task candidate
CANDIDATE_TASKS = 2000
UNCLEAR = {"unknown", "unknown speaker", "someone", "somebody", "anyone", "anybody", "nobody", "tbd"}
_WORD = re.compile(r"[a-z0-9]+")
_DATE_WORDS = {
    "today", "tomorrow", "tonight", "yesterday", "week", "weeks", "month", "months", "day", "days", "next", "end",
    "eod", "eow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "morning", "afternoon",
    "evening", "noon", "asap", "soon", "later",
}
_STOP = {
    "a", "an", "the", "and", "or", "to", "of", "for", "on", "in", "at", "by", "with", "from", "into", "about", "up",
    "i", "ll", "will", "shall", "we", "you", "he", "she", "they", "it", "me", "us", "our", "my", "your", "their",
    "his", "her", "this", "that", "these", "those", "be", "is", "are", "was", "can", "could", "should", "would",
    "going", "gonna", "need", "needs", "action", "please", "also", "then", "s", "t", "d", "ve", "re", "m", "all",
    "before", "after", "until", "over", "out", "off", "so", "just", "get", "make", "sure", "let", "lets", "do",
}


# ---------------------------------------------------------------- normalization


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold().replace("'", " ").replace("\u2019", " "))


def action_tokens(title: str, drop: set[str] = frozenset()) -> list[str]:
    """Content words of an action: no stopwords, numbers, dates or the responsible person's name; plurals folded."""
    out = []
    for word in _words(title):
        if word in _STOP or word in _DATE_WORDS or word in drop or any(c.isdigit() for c in word):
            continue
        out.append(word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word)
    return sorted(set(out))


def jaccard(a: list[str] | set[str], b: list[str] | set[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def _hash(*parts) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()


def identity(responsible_key: str, key: str, work: str | None = None) -> str:
    return _hash("meeting", responsible_key, key, work)


# ---------------------------------------------------------------- drafts


@dataclass
class Draft:
    kind: str  # create | update | link | suggestion
    title: str
    description: str = ""
    responsible: str | None = None  # actor ID
    mention: str | None = None
    collective: bool = False
    due: date | None = None
    due_at: datetime | None = None
    project_id: str | None = None
    assertion_ids: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    target: str | None = None  # existing task ID
    match: str | None = None  # stable | candidate
    work: str | None = None  # engine-named work in the identity
    follows: str | None = None  # the approved proposal a follow-up update is about
    before: dict | None = None
    after: dict | None = None
    reason: str = ""
    preselected: bool = False
    basis: str | None = None  # the action pass's owner_basis; None for older runs and bare commitments
    verified: bool | None = None  # the action pass's checker confirmed the item (None: it did not answer)

    @property
    def responsible_key(self) -> str:
        if self.responsible:
            return self.responsible
        return "mention:" + " ".join(_words(self.mention)) if self.mention else "unassigned"

    @property
    def key(self) -> str:
        drop = set(_words(self.mention or ""))
        return " ".join(action_tokens(self.title, drop)) or " ".join(_words(self.title))

    @property
    def fingerprint(self) -> str:
        if self.follows:
            return _hash("follow", self.follows)
        return identity(self.responsible_key, self.key, self.work)

    def material(self) -> str:
        return _hash(self.kind, self.responsible, self.due, self.project_id, self.target, self.collective,
                     sorted((self.after or {}).items()))

    def wording(self) -> str:
        return _hash(" ".join(_words(self.title)))


def _local_date(at: datetime | None, tz: str) -> date | None:
    return at.astimezone(ZoneInfo(tz)).date() if at else None


def _task_for(db, owner: str, entity_type: str | None, entity_id: str | None) -> str | None:
    """A task ID for an engine target: a task, or a Linear issue through its synced task mapping."""
    if not entity_id:
        return None
    if entity_type == "task":
        task = db.get(Task, entity_id)
        return task.id if task is not None and task.owner_id == owner else None
    if entity_type == "linear_issue":
        return db.scalar(select(LinearIssue.task_id).where(
            LinearIssue.owner_id == owner, (LinearIssue.id == entity_id) | (LinearIssue.remote_id == entity_id)))
    return None


def _responsible_links(db, owner: str, ids: list[str]) -> dict[str, list[tuple[str | None, str, bool]]]:
    """assertion ID -> [(actor ID or None, mention, confirmed)] from responsible links, decided ones first."""
    link = ContextEntityLink
    out: dict[str, list[tuple[str | None, str, bool]]] = {}
    rows = db.execute(select(link.assertion_id, link.target_id, link.target_type, link.status, link.mention)
                      .where(link.owner_id == owner, link.assertion_id.in_(ids),
                             link.role.in_(("responsible", "responsible_mention")))
                      .order_by(link.status != "confirmed", link.created_at, link.id)).all()
    for assertion_id, target_id, target_type, status, mention in rows:
        actor = target_id if target_type == "actor" and status in views.LINKED else None
        out.setdefault(assertion_id, []).append((actor, mention, status == "confirmed"))
    return out


def _segment_ids(db, revision_id: str) -> dict[int, str]:
    return dict(db.execute(select(ContextSegment.seq, ContextSegment.id)
                           .where(ContextSegment.revision_id == revision_id)).all())


def _assertion_evidence(db, owner: str, ids: list[str], revision_id: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    e = ContextEvidence
    for row in db.scalars(select(e).where(e.owner_id == owner, e.assertion_id.in_(ids), e.revision_id == revision_id,
                                          e.role == "support", e.quote != "").order_by(e.created_at, e.id)):
        out.setdefault(row.assertion_id, []).append(
            {"segment_id": row.segment_id, "quote": row.quote, "char_start": row.char_start, "char_end": row.char_end})
    return out


def _who(responsible: list[dict], links: list[tuple[str | None, str, bool]]) -> tuple[str | None, str | None]:
    """The responsible actor: a human-confirmed link, else the actor the engine resolved by name, else a linked
    actor. The engine's name match comes before inferred links because a recorder's speaker label links to the
    user's own actor even when someone else recorded the meeting."""
    named = next((r for r in responsible if r.get("entity_type") == "actor"), {}) if responsible else {}
    confirmed = next((a for a, _, sure in links if a and sure), None)
    actor = confirmed or named.get("entity_id") or next((a for a, _, _ in links if a), None)
    mention = named.get("mention") or next((m for _, m, _ in links if m), None)
    return actor, mention


def _drafts(db, owner: str, source: ContextSource, revision: ContextSourceRevision, actions: list[dict]) -> list[Draft]:
    tz = revision.timezone or preferences(db, owner)["timezone"]
    a = ContextAssertion
    ids = sorted({i for item in actions for i in item.get("assertion_ids") or [] if i})
    live = {r.id: r for r in db.scalars(select(a).where(
        a.owner_id == owner,
        (a.id.in_(ids)) | ((a.kind == "commitment") & exists().where(
            ContextEvidence.assertion_id == a.id, ContextEvidence.revision_id == revision.id))))}
    links = _responsible_links(db, owner, [*live, *(r.supersedes_id for r in live.values() if r.supersedes_id)])

    def linked(row: ContextAssertion) -> list[tuple[str | None, str, bool]]:
        """A Changed successor keeps its predecessor's resolved responsibility."""
        return links.get(row.id) or links.get(row.supersedes_id or "", [])

    quotes = _assertion_evidence(db, owner, list(live), revision.id)
    seq_ids = _segment_ids(db, revision.id)

    def usable(row: ContextAssertion) -> bool:
        return row.lifecycle in ("current", "contested") and row.modality == "asserted" and not row.completion_evidence

    def project_of(rows: list[ContextAssertion]) -> str | None:
        return next((r.subject_id for r in rows if r.subject_type == "project"), None)

    drafts, covered = [], set()
    for item in actions:
        action = ProposedAction.model_validate(item["action"])
        cited = [live[i] for i in item.get("assertion_ids") or [] if i in live]
        covered |= {r.id for r in cited}
        if item.get("assertion_ids") and not any(usable(r) for r in cited):
            continue  # retracted, hypothetical, completed or superseded: context only
        rows = [r for r in cited if usable(r)]
        ref = action.responsible[0] if action.responsible else None
        actor, mention = _who([r.model_dump() for r in action.responsible], [x for r in rows for x in linked(r)])
        if ref and ref.entity_type != "actor":
            actor = None
        due_at = action.due.at if action.due else next((r.due_at for r in rows if r.due_at), None)
        target = _task_for(db, owner, action.target.entity_type, action.target.entity_id) if action.target else None
        evidence = [{"segment_id": seq_ids.get(e.segment_seq), "quote": e.quote, "char_start": e.char_start,
                     "char_end": e.char_end} for e in action.evidence]
        drafts.append(Draft(
            kind=action.kind, title=action.title, description=action.description, responsible=actor, mention=mention,
            collective=bool(mention) and mention.casefold().strip() in views.COLLECTIVE, due=_local_date(due_at, tz),
            due_at=due_at, project_id=project_of(rows), assertion_ids=[r.id for r in rows], evidence=evidence,
            target=target, match="stable" if target else None,
            work=f"task:{target}" if target and action.kind in ("update", "link") else None, basis=item.get("basis"),
            verified=item.get("verified")))
    for row in live.values():
        if row.id in covered or row.kind != "commitment" or not usable(row):
            continue
        actor, mention = _who(row.responsible or [], linked(row))
        drafts.append(Draft(
            kind="create", title=row.statement[:500], responsible=actor, mention=mention,
            collective=bool(mention) and mention.casefold().strip() in views.COLLECTIVE,
            due=_local_date(row.due_at, tz), due_at=row.due_at,
            project_id=row.subject_id if row.subject_type == "project" else None, assertion_ids=[row.id],
            evidence=quotes.get(row.id, [])))
    return drafts


# ---------------------------------------------------------------- matching and recommendation


def _own_tasks(db, owner: str, source_id: str) -> set[str]:
    """Tasks this source's own proposals published or linked: never a match for its own drafts."""
    return set(db.scalars(select(ContextProposal.task_id).where(
        ContextProposal.owner_id == owner, ContextProposal.source_id == source_id,
        ContextProposal.task_id.is_not(None))))


def _stable_targets(db, owner: str, source_id: str, drafts: list[Draft], own: set[str]) -> None:
    """Fill `target` from stable links: the commitment's linked tasks, or another meeting's published proposal
    with the same identity."""
    rows = list(db.scalars(select(ContextAssertion).where(
        ContextAssertion.id.in_({i for d in drafts for i in d.assertion_ids}))))
    linked = views.linked_tasks(db, owner, rows)
    for draft in drafts:
        if draft.target or draft.kind == "suggestion":
            continue
        tasks = [t for i in draft.assertion_ids for t in linked.get(i, []) if t.id not in own and not t.is_template]
        if not tasks:
            tasks = list(db.scalars(select(Task).join(ContextProposal, ContextProposal.task_id == Task.id).where(
                ContextProposal.owner_id == owner, ContextProposal.fingerprint == draft.fingerprint,
                ContextProposal.source_id != source_id, ContextProposal.disposition == "published",
                Task.owner_id == owner).order_by(ContextProposal.created_at).limit(1)))
        if tasks:
            draft.target, draft.match = tasks[0].id, "stable"


def _candidate_targets(db, owner: str, drafts: list[Draft], own: set[str]) -> None:
    """A reversible match: the open task whose title shares most content words with the action."""
    wanted = [d for d in drafts if not d.target and d.kind in ("create", "suggestion")]
    if not wanted:
        return
    tasks = db.execute(select(Task.id, Task.title).where(
        Task.owner_id == owner, Task.archived.is_(False), Task.is_template.is_(False),
        Task.status.not_in(("completed", "cancelled"))).order_by(Task.updated_at.desc()).limit(CANDIDATE_TASKS)).all()
    titles = [(task_id, action_tokens(title)) for task_id, title in tasks if task_id not in own]
    for draft in wanted:
        words = draft.key.split()
        best = max(((jaccard(words, t), task_id) for task_id, t in titles if len(set(words) & set(t)) >= 2),
                   default=(0.0, None))
        if best[0] >= CANDIDATE:
            draft.target, draft.match = best[1], "candidate"


def _iso(value) -> str | None:
    return value.isoformat() if value else None


MINE = {
    "stated_promise": "You said you would do this, and no task covers it yet.",
    "accepted_request": "You agreed to do this when asked, and no task covers it yet.",
    "self_assigned_intent": "You said you mean to do this, and no task covers it yet.",
    "assigned_by_other": "This was handed to you in the meeting, and no task covers it yet.",
}
IMPLIED = "Implied by the discussion. Check it if it is yours to do."
UNCONFIRMED = "The checker did not confirm this one. Check it if it is yours."
INFERRED_MINE = "Likely yours, inferred from the discussion. Check it if so."


def _name(db, owner: str, draft: Draft) -> str:
    if draft.mention:
        return draft.mention
    return (views.entity_name(db, owner, "actor", draft.responsible) if draft.responsible else None) or "someone"


def _recommend(db, owner: str, draft: Draft, me: str | None) -> Draft:
    if draft.kind == "suggestion" and draft.match != "candidate":
        draft.target = draft.match = None
        draft.reason = "A useful next step, but nobody promised it."
        return draft
    if draft.match == "candidate":
        draft.kind = "link"
        draft.reason = "This looks like an existing task. Link it if it is the same work."
        return draft
    if draft.target:
        task = db.get(Task, draft.target)
        after = {}
        if draft.due and draft.due != task.due_date:
            after["due_date"] = draft.due.isoformat()
        if draft.responsible and draft.responsible != task.assignee_id:
            after["assignee_id"] = draft.responsible
        if after:
            draft.kind = "update"
            draft.before = {"due_date": _iso(task.due_date), "assignee_id": task.assignee_id,
                            "revision": task.revision, "title": task.title}
            draft.after = after
            draft.reason = "The meeting changed work that already has a task."
        else:
            draft.kind = "link"
            draft.reason = "An existing task already covers this."
        return draft
    draft.kind = "create"
    unclear = not (draft.responsible or draft.mention) or (draft.mention or "").casefold().strip() in UNCLEAR
    if draft.collective or (unclear and not draft.responsible):
        draft.reason = "Who owns this is unclear."
    elif draft.basis == "inferred":  # a likely owner the lines can't prove (unseparated speakers): never checked
        draft.reason = INFERRED_MINE if draft.responsible and draft.responsible == me else \
            f"Likely {_name(db, owner, draft)}, inferred from the discussion."
    elif draft.responsible and draft.responsible == me:
        # Action-pass items are checked only when the checker confirmed them; older runs carry no basis.
        unconfirmed = draft.basis is not None and draft.verified is not True
        draft.preselected = draft.basis != "implied_by_discussion" and not unconfirmed
        draft.reason = IMPLIED if draft.basis == "implied_by_discussion" else UNCONFIRMED if unconfirmed else \
            MINE.get(draft.basis or "", MINE["stated_promise"])
    else:
        draft.reason = f"Waiting on {_name(db, owner, draft)}. Track it only if you need to."
    return draft


def _merge(drafts: list[Draft]) -> list[Draft]:
    """One draft per identity and per targeted task (chunk overlap and duplicate actions)."""
    out: dict[str, Draft] = {}
    for draft in drafts:
        key = f"task:{draft.target}:{draft.kind}" if draft.target and draft.match == "stable" else draft.fingerprint
        kept = out.get(key)
        if kept is None:
            out[key] = draft
            continue
        kept.assertion_ids = list(dict.fromkeys(kept.assertion_ids + draft.assertion_ids))
        seen = {(e.get("segment_id"), e.get("quote")) for e in kept.evidence}
        kept.evidence += [e for e in draft.evidence if (e.get("segment_id"), e.get("quote")) not in seen]
        kept.due, kept.due_at = kept.due or draft.due, kept.due_at or draft.due_at
        kept.project_id = kept.project_id or draft.project_id
        kept.description = kept.description or draft.description
    return list(out.values())


# ---------------------------------------------------------------- persistence


def destination(db, owner: str) -> str:
    """The owner's remembered destination choice, Eridani by default."""
    state = db.get(ContextOwnerState, owner)
    chosen = ((state.settings or {}) if state else {}).get("review_destination")
    return chosen if chosen in DESTINATIONS else "eridani"


def _payload(draft: Draft, source: ContextSource, revision: ContextSourceRevision, db, explanation=None) -> dict:
    target = None
    if draft.target:
        task = db.get(Task, draft.target)
        target = {"task_id": task.id, "title": task.title, "revision": task.revision}
    return {"origin": "meeting", "key": draft.key, "responsible_key": draft.responsible_key, "work": draft.work,
            "follows": draft.follows, "description": draft.description, "assignee_id": draft.responsible,
            "responsible_mention": draft.mention, "collective": draft.collective, "project_id": draft.project_id,
            "due_date": _iso(draft.due), "target": target, "match": draft.match, "before": draft.before,
            "after": draft.after, "reason": draft.reason, "explanation": explanation,
            "material": draft.material(), "wording": draft.wording(), "source_title": source.title,
            "occurred_at": _iso(revision.occurred_at)}


def _explain(old: dict, draft: Draft) -> str:
    notes = []
    if old.get("due_date") != _iso(draft.due):
        notes.append(f"Due date changed from {old.get('due_date') or 'none'} to {_iso(draft.due) or 'none'}.")
    if old.get("assignee_id") != draft.responsible:
        notes.append("The person responsible changed.")
    if (old.get("target") or {}).get("task_id") != draft.target:
        notes.append("It now matches an existing task." if draft.target else "It no longer matches a task.")
    if old.get("project_id") != draft.project_id:
        notes.append("The project changed.")
    if old.get("after") != draft.after and draft.after:
        notes.append("The proposed change to the task differs.")
    if not notes and old.get("wording") != draft.wording():
        notes.append("The wording changed.")
    return " ".join(notes) or "The meeting record changed."


def _write(row: ContextProposal, draft: Draft, payload: dict, revision_id: str, *, bump: bool, at: datetime) -> None:
    row.kind, row.title, row.payload = draft.kind, draft.title[:500], payload
    row.responsible, row.due_at, row.revision_id = draft.responsible, draft.due_at, revision_id
    row.assertion_ids, row.evidence, row.preselected = draft.assertion_ids, draft.evidence[:10], draft.preselected
    if bump:
        row.revision += 1
    row.updated_at = at


def _follow_up(db, owner: str, row: ContextProposal, draft: Draft, me: str | None) -> Draft | None:
    """A changed draft for approved or published work: an update against the published task, when it differs."""
    if not row.task_id or db.get(Task, row.task_id) is None:
        return None
    follow = replace(draft, kind="create", target=row.task_id, match="stable", follows=row.id, before=None,
                     after=None, preselected=False, work=None)
    follow = _recommend(db, owner, follow, me)
    return follow if follow.kind == "update" else None


def _upsert(db, owner: str, source: ContextSource, revision: ContextSourceRevision, drafts: list[Draft],
            me: str | None, at: datetime) -> tuple[dict, set[str]]:
    counts = {"created": 0, "revised": 0, "unchanged": 0, "reopened": 0, "follow_ups": 0}
    p = ContextProposal
    rows = list(db.scalars(select(p).where(p.owner_id == owner, p.source_id == source.id)
                           .order_by(p.created_at, p.id).with_for_update()))
    used: set[str] = set()
    default = destination(db, owner)

    def find(draft: Draft) -> ContextProposal | None:
        pool = [r for r in rows if r.id not in used and (r.payload or {}).get("follows") == draft.follows]
        exact = next((r for r in pool if r.fingerprint == draft.fingerprint), None)
        if exact or draft.follows:
            return exact
        same = next((r for r in pool if (r.payload or {}).get("key") == draft.key), None)
        if same:
            return same
        near = [(jaccard(draft.key.split(), (r.payload or {}).get("key", "").split()), r) for r in pool
                if (r.payload or {}).get("responsible_key") == draft.responsible_key]
        score, best = max(near, key=lambda x: x[0], default=(0.0, None))
        return best if score >= NEAR else None

    queue = list(drafts)
    while queue:
        draft = queue.pop(0)
        row = find(draft)
        if row is None:
            row = p(owner_id=owner, source_id=source.id, fingerprint=draft.fingerprint, destination=default,
                    created_at=at, revision=1, disposition="pending")
            _write(row, draft, _payload(draft, source, revision, db), revision.id, bump=False, at=at)
            db.add(row)
            db.flush()
            rows.append(row)
            used.add(row.id)
            counts["created"] += 1
            emit(db, owner, "context.changed", row.id, row.revision)
            continue
        used.add(row.id)
        old = row.payload or {}
        material, wording = old.get("material") != draft.material(), old.get("wording") != draft.wording()
        if row.disposition in LOCKED:
            if material and (follow := _follow_up(db, owner, row, draft, me)):
                queue.append(follow)
                counts["follow_ups"] += 1
            else:
                counts["unchanged"] += 1
            continue
        if row.disposition == "dismissed" and not material:
            counts["unchanged"] += 1
            continue
        if not (material or wording) and row.disposition != "superseded":
            # Same work, same payload: refresh evidence without a new revision (no approval goes stale).
            row.assertion_ids = list(dict.fromkeys(row.assertion_ids + draft.assertion_ids))
            row.evidence, row.revision_id, row.updated_at = draft.evidence[:10], revision.id, at
            counts["unchanged"] += 1
            continue
        explanation = _explain(old, draft) if row.disposition != "superseded" else "It is back in the meeting record."
        reopened = row.disposition in ("dismissed", "superseded")
        _write(row, draft, _payload(draft, source, revision, db, explanation), revision.id, bump=True, at=at)
        if reopened:
            row.disposition = "pending"
            counts["reopened"] += 1
        else:
            counts["revised"] += 1
        emit(db, owner, "context.changed", row.id, row.revision)
    return counts, used


def _bases(run: ContextRun | None) -> dict[int, dict]:
    """action index -> {basis, verified} from the extract run's action-pass diagnostics."""
    result = ((run.stage_results or {}).get("result") or {}) if run else {}
    items = ((result.get("diagnostics") or {}).get("actions") or {}).get("items") or []
    return {i["action_index"]: {"basis": i["owner_basis"], "verified": i.get("verified")} for i in items
            if isinstance(i.get("action_index"), int) and i.get("owner_basis")}


def generate(db, owner: str, source_id: str, revision_id: str, *, complete: bool | None = None) -> dict:
    """Create or revise the source's proposals from the revision's extraction (see the module docstring).
    `complete` (default: the extract run succeeded) allows superseding proposals the revision no longer supports."""
    source = db.get(ContextSource, source_id)
    revision = db.get(ContextSourceRevision, revision_id)
    if source is None or revision is None or source.owner_id != owner or source.status == "deleted":
        return {"skipped": "source unavailable"}
    run = db.scalar(select(ContextRun).where(
        ContextRun.owner_id == owner, ContextRun.stage == "extract", ContextRun.revision_id == revision_id,
        ContextRun.status.in_(("succeeded", "partial"))).order_by(ContextRun.finished_at.desc().nulls_last(),
                                                                 ContextRun.created_at.desc()).limit(1))
    actions = list((run.stage_results or {}).get("actions") or []) if run else []
    bases = _bases(run)
    actions = [{**item, **bases.get(i, {})} for i, item in enumerate(actions)]
    if complete is None:
        complete = run is not None and run.status == "succeeded"
    me = views.me(db, owner)
    drafts = _drafts(db, owner, source, revision, actions)
    own = _own_tasks(db, owner, source_id)
    _stable_targets(db, owner, source_id, drafts, own)
    _candidate_targets(db, owner, drafts, own)
    drafts = _merge([_recommend(db, owner, d, me) for d in drafts])
    at = now()
    counts, used = _upsert(db, owner, source, revision, drafts, me, at)
    superseded = 0
    if complete:
        for row in db.scalars(select(ContextProposal).where(
                ContextProposal.owner_id == owner, ContextProposal.source_id == source_id,
                ContextProposal.id.not_in(used or {""}), ContextProposal.disposition.in_(OPEN)).with_for_update()):
            row.disposition, row.revision, row.updated_at = "superseded", row.revision + 1, at
            row.payload = {**(row.payload or {}), "explanation": "The meeting record no longer supports this."}
            emit(db, owner, "context.changed", row.id, row.revision)
            superseded += 1
    db.flush()
    return {**counts, "superseded": superseded, "proposals": len(used)}


# ---------------------------------------------------------------- corrections


def invalidate(db, owner: str, assertion_id: str, explanation: str) -> list[str]:
    """After Wrong: open proposals resting only on retracted interpretations are superseded. Approved and
    published work is never changed."""
    p, at, out = ContextProposal, now(), []
    for row in db.scalars(select(p).where(p.owner_id == owner, p.assertion_ids.contains([assertion_id]),
                                          p.disposition.in_((*OPEN, "dismissed"))).with_for_update()):
        live = db.scalar(select(ContextAssertion.id).where(
            ContextAssertion.id.in_(row.assertion_ids), ContextAssertion.lifecycle.in_(("current", "contested")))
            .limit(1))
        if live:
            continue
        row.disposition, row.revision, row.updated_at = "superseded", row.revision + 1, at
        row.payload = {**(row.payload or {}), "explanation": explanation}
        emit(db, owner, "context.changed", row.id, row.revision)
        out.append(row.id)
    return out


def regenerate(db, owner: str, source_id: str) -> dict:
    """Re-derive the source's proposals from its current revision (after a human correction), under the owner
    lock like the pipeline's generation."""
    apply.lock(db, owner)
    source = db.get(ContextSource, source_id)
    revision_id = db.scalar(select(ContextSourceRevision.id).where(
        ContextSourceRevision.source_id == source_id,
        ContextSourceRevision.revision == source.current_revision)) if source else None
    return generate(db, owner, source_id, revision_id, complete=False) if revision_id else {"skipped": "no revision"}


# ---------------------------------------------------------------- job


def perform(work) -> dict:
    """The `context_proposals` job {source_id, revision_id, extract_run_id?}: generate under the owner lock,
    refusing stale work like every automatic stage (pause, epoch, newer revision, deleted source)."""
    from ..db import session_scope
    from . import pipeline

    source_id, revision_id = work.payload.get("source_id"), work.payload.get("revision_id")
    if not (source_id and revision_id):
        return {"skipped": "nothing to do"}
    with session_scope() as db:
        run = pipeline._run(db, work, "proposals", source_id=source_id, revision_id=revision_id)
        if replay := pipeline._replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run, source_id=source_id, revision_id=revision_id):
            return pipeline._stale(db, work, run, reason)
        return pipeline._finish(run, summary=generate(db, work.owner, source_id, revision_id))


# ---------------------------------------------------------------- review decisions


def _open(db, owner: str, proposal_id: str) -> None:
    row = db.get(ContextProposal, proposal_id)
    if row is not None and row.owner_id == owner and row.disposition in LOCKED:
        raise DomainError("REVISION_CONFLICT", "This proposal is already approved.", 409)


def dismiss(db, owner: str, proposal_id: str, expected_revision: int, *, command_id: str, principal: str,
            reason: str = ""):
    """Finish without this proposal: the context stays; unchanged evidence never brings it back."""
    _open(db, owner, proposal_id)
    return decisions.dismiss(db, owner, proposal_id, expected_revision, command_id=command_id, principal=principal,
                             reason=reason)


def later(db, owner: str, proposal_id: str, expected_revision: int, *, command_id: str, principal: str,
          until: datetime | None = None, reason: str = ""):
    """Remind me later: pending for the daily digest (until `until`, when given)."""
    _open(db, owner, proposal_id)
    return decisions.later(db, owner, proposal_id, expected_revision, command_id=command_id, principal=principal,
                           until=until, reason=reason)
