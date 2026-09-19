"""Canonical structured retrieval plus a separate, non-authoritative semantic lane."""

import re
import time
from functools import lru_cache
from uuid import uuid4
from sqlalchemy import func, select
from . import access
from .config import get_settings
from .db import session_scope
from .domain import DomainError
from .memory_learning import EMBEDDING_MODEL, cosine, embeddings
from .search_index import snapshot, queue_index
from .search_models import SearchDocument, SearchIndexState, SearchAlias, SearchSession
from .search_schema import SearchQuery

STOP = set(
    "a an the me my please show find search get give all any for of to in on with about and or that this these those task tasks note notes record records company".split()
)


@lru_cache(maxsize=128)
def query_vector(owner, query, model, period, provider):
    # Short-lived, owner-scoped reuse across result pages. Authorization and
    # canonical records are still checked afresh before and after this lookup.
    return provider(owner, [query], 3)[0]


def normalize(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def terms(value):
    return set(normalize(value).split()) - STOP


def contains(text, phrase):
    return bool(phrase and (" " + phrase + " ") in (" " + normalize(text) + " "))


def score(query, doc, vector, stored, fts=0):
    words = terms(query)
    lexical = len(words & terms(doc["text"])) / max(1, len(words))
    similarity = (
        max((cosine(vector, v) for v in stored.vectors), default=0)
        if vector
        and stored
        and stored.fingerprint == doc["fingerprint"]
        and stored.embedding_model == EMBEDDING_MODEL
        else 0
    )
    return max(lexical * 0.7, similarity) + min(fts, 0.2), lexical, similarity


def snippet(text, query):
    positions = [text.casefold().find(w) for w in terms(query) if w in text.casefold()]
    start = max(0, min(positions, default=0) - 90)
    return ("…" if start else "") + text[start : start + 420] + ("…" if len(text) > start + 420 else "")


def permitted(db, owner, account):
    access.role(db, owner, account)
    access.authorize_execution(db, owner)
    from .bot_access import current_id, authorize

    if current_id():
        authorize(db, owner, "records:read")


def search(
    owner,
    account,
    arguments,
    *,
    conversation_id=None,
    work_id=None,
    request_key=None,
    track=True,
    note_filters=None,
):
    args = arguments if isinstance(arguments, SearchQuery) else SearchQuery.model_validate(arguments)
    enabled = get_settings().semantic_search_enabled
    with session_scope() as db:
        permitted(db, owner, account)
        # Canonical sources are authoritative, even before indexing finishes.
        documents, records = snapshot(db, owner)
        stored = {
            d.target_key: d
            for d in db.scalars(select(SearchDocument).where(SearchDocument.owner_id == owner))
        }
        if enabled:
            queue_index(db, owner, dirty=False)
    vector, fallback = None, not enabled
    if enabled and any(d.vectors for d in stored.values()):
        try:
            vector = query_vector(
                owner, args.query, EMBEDDING_MODEL, int(time.monotonic() // 120), embeddings
            )
        except Exception:
            fallback = True
    # Refresh after the provider call. Deleted/reassigned records and revoked access win.
    with session_scope() as db:
        permitted(db, owner, account)
        documents, records = snapshot(db, owner)
        state = db.get(SearchIndexState, owner)
        missing = sum(
            key not in stored
            or stored[key].fingerprint != d["fingerprint"]
            or stored[key].embedding_model != EMBEDDING_MODEL
            or not stored[key].vectors
            for key, d in documents.items()
        )
        aliases = (
            list(
                db.scalars(
                    select(SearchAlias).where(
                        SearchAlias.owner_id == owner,
                        SearchAlias.account_id == account,
                        SearchAlias.status.in_(["provisional", "confirmed"]),
                    )
                )
            )
            if enabled
            else []
        )
        aliases = [
            a
            for a in aliases
            if a.target_key in documents
            and a.target_fingerprint == documents[a.target_key]["definition_fingerprint"]
            and contains(args.query, a.phrase)
        ]
        tsq = func.websearch_to_tsquery("english", args.query)
        tsv = func.to_tsvector("english", SearchDocument.content)
        fts = dict(
            db.execute(
                select(SearchDocument.target_key, func.ts_rank_cd(tsv, tsq)).where(
                    SearchDocument.owner_id == owner, tsv.op("@@")(tsq)
                )
            ).all()
        )
        candidates = []
        for key, doc in documents.items():
            value, lexical, semantic = score(args.query, doc, vector, stored.get(key), fts.get(key, 0))
            exact = contains(args.query, normalize(doc["label"])) and bool(terms(doc["label"]))
            matches = [a for a in aliases if a.target_key == key]
            value += 1.5 if exact else 0.95 if matches else 0
            if not exact and not matches and lexical == 0 and semantic < 0.35:
                continue
            candidates.append(
                dict(
                    key=key,
                    kind=doc["kind"],
                    label=doc["label"],
                    type_id=doc.get("type_id"),
                    score=round(value, 4),
                    exact=exact,
                    alias=bool(matches),
                    provisional=bool(matches and all(a.status != "confirmed" for a in matches)),
                    evidence=snippet(doc["text"], args.query),
                )
            )
        candidates.sort(key=lambda c: (-c["score"], c["key"]))
        resolved = list(dict.fromkeys(args.resolved))
        for key in resolved:
            if key not in documents:
                raise DomainError(
                    "INVALID_SEARCH_TARGET",
                    "That search target is unavailable. Search again for its current identity.",
                )
        if not resolved:
            named = [c for c in candidates if c["exact"] and c["kind"] == "record"]
            learned = [c for c in candidates if c["alias"]]
            options = named or learned
            if len(options) == 1:
                resolved = [options[0]["key"]]
        type_ids = {d["type_id"] for d in documents.values() if d["kind"] == "type"}
        if set(args.type_ids) - type_ids:
            raise DomainError("INVALID_SEARCH_FILTER", "Choose existing record types.")
        if args.home_id and args.home_id not in records:
            raise DomainError("INVALID_SEARCH_FILTER", "That main home is unavailable.")
        field_ids = {d["field_id"] for d in documents.values() if d["kind"] == "field"}
        if set(args.values) - field_ids:
            raise DomainError("INVALID_SEARCH_FILTER", "Choose existing fields.")
        note_ids = None
        if note_filters is not None:
            from .notes import scope_notes
            from .models import Note, NoteProjectLink, NoteTaskLink
            from sqlalchemy import or_

            q = scope_notes(
                select(Note.id).where(Note.owner_id == owner, Note.archived.is_(False)),
                note_filters.get("space_id"),
                note_filters.get("area_id"),
                note_filters.get("goal_id"),
            )
            if note_filters.get("project_id"):
                project = note_filters["project_id"]
                q = q.where(
                    or_(
                        Note.project_id == project,
                        Note.id.in_(
                            select(NoteProjectLink.note_id).where(NoteProjectLink.project_id == project)
                        ),
                    )
                )
            if note_filters.get("task_id"):
                q = q.where(
                    Note.id.in_(
                        select(NoteTaskLink.note_id).where(
                            NoteTaskLink.task_id == note_filters["task_id"], NoteTaskLink.linked.is_(True)
                        )
                    )
                )
            from .note_lists import filtered
            q = filtered(db, owner, q, note_filters.get("list_id"), note_filters.get("uncategorized", False))
            note_ids = set(db.scalars(q))
        structured, possible = [], []
        for identity, data in records.items():
            r = data["record"]
            if note_ids is not None and r.get("note_id") not in note_ids:
                continue
            if (
                bool(r["archived"]) != args.archived
                or (args.capability and args.capability not in r["capabilities"])
                or (args.type_ids and r["type_id"] not in args.type_ids)
            ):
                continue
            if (
                args.home_id
                and args.home_id not in {p["id"] for p in r["home"]}
                and r["parent_id"] != args.home_id
            ):
                continue
            if args.statuses and r["status_meaning"] not in args.statuses:
                continue
            if any(
                r["values"].get(k) != v and not (isinstance(r["values"].get(k), list) and v in r["values"][k])
                for k, v in args.values.items()
            ):
                continue
            due = data["bindings"].get("due_date")
            if (args.due_from and (not due or due < args.due_from.isoformat())) or (
                args.due_through and (not due or due > args.due_through.isoformat())
            ):
                continue
            key = "record:" + identity
            doc = documents.get(key) or dict(text=r["title"] + "\n" + (r["body"] or ""), fingerprint="")
            value, lexical, semantic = score(args.query, doc, vector, stored.get(key), fts.get(key, 0))
            matched = [k for k in resolved if k == key or k in data["targets"]]
            explicit = bool(resolved and len(matched) == len(resolved))
            if args.strict and not explicit:
                continue
            if not explicit and lexical == 0 and semantic < 0.35:
                continue
            evidence = snippet(doc["text"], args.query)
            short = {**r, "body": (r["body"] or "")[:600]}
            item = dict(
                record=short,
                score=round(value, 4),
                evidence=evidence,
                match="structured" if explicit else "possible",
                matched_targets=matched,
                reason="Saved fields, main home or relationships match the selected identity."
                if explicit
                else "Relevant wording or meaning; this is not a saved classification.",
                current_assignment=dict(home=r["home"], fields=data["field_labels"]),
            )
            (structured if explicit else possible).append(item)
        for group in (structured, possible):
            group.sort(key=lambda m: (-m["score"], m["record"]["id"]))
        top_candidates = candidates[:8]
        # Explicit refinements remain available to the selection tool even beyond the first page.
        for key in resolved:
            if not any(c["key"] == key for c in top_candidates):
                d = documents[key]
                top_candidates.append(
                    dict(
                        key=key,
                        kind=d["kind"],
                        label=d["label"],
                        type_id=d.get("type_id"),
                        exact=True,
                        score=1,
                        evidence=snippet(d["text"], args.query),
                    )
                )
        page = lambda group: dict(
            items=group[args.offset : args.offset + args.limit],
            total=len(group),
            next_offset=args.offset + args.limit if len(group) > args.offset + args.limit else None,
        )
        referring_text = args.query
        if work_id and track:
            from .models import AgentWork
            from .work_crypto import unseal

            work = db.get(AgentWork, work_id)
            if work and work.owner_id == owner and work.account_id == account:
                referring_text = unseal(work.input_ciphertext).get("message", args.query)
        result = dict(
            query=args.query,
            referring_text=referring_text,
            enabled=enabled,
            mode="keyword_fallback" if fallback or vector is None else "hybrid",
            resolved=resolved,
            resolutions=top_candidates,
            ambiguous=not resolved and len(top_candidates) > 1,
            structured=page(structured),
            possible=page(possible),
            index=dict(
                status=state.status if state else "not_started", documents=len(documents), pending=missing
            ),
            search_id=None,
        )
        from .bot_access import current_id

        from .search_learning import learning_enabled

        if enabled and track and not current_id() and learning_enabled(db, owner, account):
            key = request_key or str(uuid4())
            prior = db.scalar(
                select(SearchSession).where(
                    SearchSession.owner_id == owner,
                    SearchSession.account_id == account,
                    SearchSession.request_key == key,
                )
            )
            if not prior:
                prior = SearchSession(
                    owner_id=owner,
                    account_id=account,
                    request_key=key,
                    query=referring_text,
                    conversation_id=conversation_id,
                    work_id=work_id,
                )
                db.add(prior)
                db.flush()
            prior.candidates = [c["key"] for c in top_candidates]
            prior.result_ids = list(
                dict.fromkeys(
                    m["record"]["id"] for m in result["structured"]["items"] + result["possible"]["items"]
                )
            )
            result["search_id"] = prior.id
        return result
