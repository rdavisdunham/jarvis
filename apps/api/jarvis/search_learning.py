"""Search vocabulary is weak, reversible evidence, never a personal fact or label."""

import re
from datetime import timedelta
from sqlalchemy import select
from .models import Conversation, Job, AgentWork, now
from .search_models import SearchAlias, SearchSession, SearchPreference
from .search_index import snapshot, digest
from .search_service import normalize, contains, permitted
from .domain import DomainError, advisory, check_revision, serial


def learning_enabled(db, owner, account):
    preference = db.get(SearchPreference, (owner, account))
    return preference.learning if preference else True


def require_session(db, owner, account, identity):
    permitted(db, owner, account)
    advisory(db, "search-alias:" + owner + ":" + account)
    row = db.get(SearchSession, identity, with_for_update=True)
    if not row or row.owner_id != owner or row.account_id != account:
        raise DomainError("NOT_FOUND", "This search is unavailable.", 404)
    return row


def choose(db, owner, account, args):
    row = require_session(db, owner, account, args.search_id)
    docs, _ = snapshot(db, owner)
    if args.target_key not in row.candidates or args.target_key not in docs:
        raise DomainError("INVALID_SEARCH_TARGET", "Choose an identity returned by this search.")
    phrase = normalize(args.phrase)
    if not contains(row.query, phrase) or len(phrase) < 2:
        raise DomainError("INVALID_SEARCH_PHRASE", "Copy the referring phrase from the original search.")
    if set(args.record_ids) - set(row.result_ids):
        raise DomainError("INVALID_SEARCH_TARGET", "Choose records returned by this search.")
    if row.selected_key and (row.selected_key != args.target_key or row.phrase != phrase):
        raise DomainError(
            "SEARCH_ALREADY_SELECTED",
            "Use search_feedback to correct this search, or run a refined search.",
            409,
        )
    row.selected_key, row.phrase, row.selected_records = (
        args.target_key,
        phrase,
        list(dict.fromkeys(args.record_ids)),
    )
    return {"search_id": row.id, "selected": args.target_key, "learning": "waiting_for_use_or_continuation"}


def accept_search(db, row, signal):
    if (
        row.suppressed
        or row.outcome in ("accepted", "corrected")
        or not row.presented_at
        or not row.selected_key
        or not row.phrase
        or not learning_enabled(db, row.owner_id, row.account_id)
    ):
        return
    docs, _ = snapshot(db, row.owner_id)
    target = docs.get(row.selected_key)
    if not target:
        return
    advisory(db, "search-alias:" + row.owner_id + ":" + row.account_id)
    alias = db.scalar(
        select(SearchAlias).where(
            SearchAlias.owner_id == row.owner_id,
            SearchAlias.account_id == row.account_id,
            SearchAlias.phrase == row.phrase,
            SearchAlias.target_key == row.selected_key,
        )
    )
    if alias and alias.status in ("forgotten", "paused", "conflict"):
        row.suppressed = True
        return
    if not alias:
        alias = SearchAlias(
            owner_id=row.owner_id,
            account_id=row.account_id,
            phrase=row.phrase,
            target_key=row.selected_key,
            target_fingerprint=target["definition_fingerprint"],
        )
        db.add(alias)
        db.flush()
    if alias.target_fingerprint != target["definition_fingerprint"]:
        alias.status = "paused"
        return
    alias.updated_at = now()
    row.alias_id, row.outcome, row.signal, row.accepted_at = alias.id, "accepted", signal, now()


def reject_search(db, row, signal):
    if row.outcome == "corrected":
        return
    row.outcome, row.signal = "corrected", signal
    if row.alias_id:
        alias = db.get(SearchAlias, row.alias_id)
        if alias and alias.status != "forgotten":
            alias.status = "conflict"
            alias.revision += 1
            alias.updated_at = now()


def feedback(db, owner, account, args):
    row = require_session(db, owner, account, args.search_id)
    if args.outcome == "corrected":
        reject_search(db, row, "correction")
        if args.replacement_key:
            docs, _ = snapshot(db, owner)
            if args.replacement_key not in docs:
                raise DomainError("INVALID_SEARCH_TARGET", "Search for the corrected identity first.")
            row.selected_key = args.replacement_key
            row.alias_id = None
            row.outcome = "unknown"
            # This tool is a correction on a presented search, not proof of presentation.
            accept_search(db, row, "explicit_correction")
    else:
        accept_search(db, row, "explicit_acceptance")
    return {"search_id": row.id, "outcome": row.outcome}


def event(db, owner, account, args):
    permitted(db, owner, account)
    q = select(SearchSession).where(SearchSession.owner_id == owner, SearchSession.account_id == account)
    if args.search_id:
        q = q.where(SearchSession.id == args.search_id)
    elif args.work_id:
        work = db.get(AgentWork, args.work_id)
        from .work_continuation import latest

        work = latest(db, work) if work else None
        job = db.get(Job, work.id) if work else None
        if (
            not work
            or work.owner_id != owner
            or work.account_id != account
            or not job
            or job.status not in ("succeeded", "needs_input", "partial")
        ):
            return {"recorded": False}
        q = q.where(SearchSession.work_id == work.id)
    else:
        raise DomainError("INVALID_ARGUMENT", "Identify the presented search or response.")
    advisory(db, "search-alias:" + owner + ":" + account)
    rows = list(db.scalars(q.with_for_update()))
    for row in rows:
        if not row.selected_key or row.suppressed:
            continue
        if args.kind == "used" and (not args.record_id or args.record_id not in row.selected_records):
            continue
        if not row.presented_at:
            row.presented_at = now()
        if args.kind == "used":
            accept_search(db, row, "used:" + args.record_id)
    return {"recorded": bool(rows)}


CORRECTION = re.compile(
    r"\b(wrong|incorrect|i meant|not (?:that|this|the|what)|(?:different|another) (?:one|result|client|project|field)|that (?:isn t|wasn t)|no (?:i|that|the|it))\b",
    re.I,
)


def observe_source(db, source):
    """Real transcript/UI presentation only; background tool output is never acceptance."""
    if not source.conversation_id:
        return
    conversation = db.get(Conversation, source.conversation_id)
    if not conversation:
        return
    from .models import AuthSession
    from .access import role

    session = db.scalar(
        select(AuthSession).where(
            AuthSession.device_id == conversation.device_id, AuthSession.expires_at > now()
        )
    )
    if not session or (session.workspace_id or session.owner_id) != source.owner_id:
        return
    account = session.owner_id
    try:
        role(db, source.owner_id, account)
    except DomainError:
        return
    advisory(db, "search-alias:" + source.owner_id + ":" + account)
    q = select(SearchSession).where(
        SearchSession.owner_id == source.owner_id,
        SearchSession.account_id == account,
        SearchSession.conversation_id == source.conversation_id,
        SearchSession.selected_key.is_not(None),
        SearchSession.suppressed.is_(False),
    )
    if source.role == "assistant" and source.native_id.startswith("live:"):
        docs, _ = snapshot(db, source.owner_id)
        for row in db.scalars(
            q.where(SearchSession.presented_at.is_(None), SearchSession.created_at <= source.created_at)
        ):
            target = docs.get(row.selected_key)
            # A spoken target anchors this interpretation. "Here it is" alone stays unknown.
            if target and contains(source.content, normalize(target["label"])):
                row.presented_at = source.created_at
        return
    if source.role != "user":
        return
    row = db.scalar(
        q.where(
            SearchSession.presented_at.is_not(None),
            SearchSession.presented_at < source.created_at,
            SearchSession.presented_at >= source.created_at - timedelta(minutes=15),
        )
        .order_by(SearchSession.presented_at.desc())
        .limit(1)
    )
    if not row or not normalize(source.content) or normalize(source.content) in ("eri", "hey eri"):
        return
    if row.signal == "possible_correction:" + source.id or row.signal == "continued:" + source.id:
        return
    if CORRECTION.search(normalize(source.content)) or normalize(source.content) == "no":
        # Conservative: an ambiguous correction never increases confidence.
        reject_search(db, row, "possible_correction:" + source.id)
    else:
        accept_search(db, row, "continued:" + source.id)


def context(db, owner, account, conversation_id):
    rows = list(
        db.scalars(
            select(SearchSession)
            .where(
                SearchSession.owner_id == owner,
                SearchSession.account_id == account,
                SearchSession.conversation_id == conversation_id,
                SearchSession.selected_key.is_not(None),
                SearchSession.presented_at.is_not(None),
                SearchSession.suppressed.is_(False),
            )
            .order_by(SearchSession.presented_at.desc())
            .limit(3)
        )
    )
    return [
        dict(
            search_id=r.id,
            query=r.query,
            phrase=r.phrase,
            selected_key=r.selected_key,
            record_ids=r.selected_records,
            outcome=r.outcome,
        )
        for r in rows
    ]


def state(db, owner, account):
    permitted(db, owner, account)
    docs, _ = snapshot(db, owner)
    rows = list(
        db.scalars(
            select(SearchAlias)
            .where(
                SearchAlias.owner_id == owner,
                SearchAlias.account_id == account,
                SearchAlias.status != "forgotten",
            )
            .order_by(SearchAlias.updated_at.desc())
        )
    )
    evidence = list(
        db.scalars(
            select(SearchSession).where(
                SearchSession.owner_id == owner,
                SearchSession.account_id == account,
                SearchSession.alias_id.in_([a.id for a in rows]),
                SearchSession.suppressed.is_(False),
            )
        )
    )
    items = []
    for alias in rows:
        target = docs.get(alias.target_key)
        history = [s for s in evidence if s.alias_id == alias.id]
        items.append(
            {
                **serial(alias),
                "label": target["label"] if target else "Unavailable record/field",
                "available": bool(target and target["definition_fingerprint"] == alias.target_fingerprint),
                "positive_count": sum(s.outcome == "accepted" for s in history),
                "sources": [
                    dict(
                        search_id=s.id,
                        query=s.query,
                        signal=s.signal,
                        outcome=s.outcome,
                        created_at=s.created_at.isoformat(),
                    )
                    for s in history[-10:]
                ],
            }
        )
    return {
        "learning": learning_enabled(db, owner, account),
        "items": items,
        "targets": [dict(key=d["key"], label=d["label"], kind=d["kind"]) for d in docs.values()],
    }


def change_alias(db, owner, account, identity, args):
    permitted(db, owner, account)
    advisory(db, "search-alias:" + owner + ":" + account)
    alias = db.get(SearchAlias, identity, with_for_update=True)
    if not alias or alias.owner_id != owner or alias.account_id != account:
        raise DomainError("NOT_FOUND", "This alias is unavailable.", 404)
    check_revision(alias, args.expected_revision)
    docs, _ = snapshot(db, owner)
    if args.action == "forget":
        alias.status = "forgotten"
        for source in db.scalars(select(SearchSession).where(SearchSession.alias_id == alias.id)):
            source.suppressed = True
    elif args.action == "pause":
        alias.status = "paused"
    else:
        key = args.target_key if args.action == "correct" else alias.target_key
        if key not in docs:
            raise DomainError("INVALID_SEARCH_TARGET", "Choose an available identity.")
        if args.action == "correct" and key != alias.target_key:
            # Preserve the rejected mapping as a tombstone rather than silently overwriting evidence.
            alias.status = "forgotten"
            for source in db.scalars(select(SearchSession).where(SearchSession.alias_id == alias.id)):
                source.suppressed = True
            replacement = db.scalar(
                select(SearchAlias).where(
                    SearchAlias.owner_id == owner,
                    SearchAlias.account_id == account,
                    SearchAlias.phrase == alias.phrase,
                    SearchAlias.target_key == key,
                )
            )
            if not replacement:
                replacement = SearchAlias(
                    owner_id=owner,
                    account_id=account,
                    phrase=alias.phrase,
                    target_key=key,
                    target_fingerprint=docs[key]["definition_fingerprint"],
                )
                db.add(replacement)
                db.flush()
            replacement.status = "confirmed"
            replacement.target_fingerprint = docs[key]["definition_fingerprint"]
            replacement.revision += 1
            replacement.updated_at = now()
        else:
            alias.status = "confirmed"
            alias.target_fingerprint = docs[key]["definition_fingerprint"]
    if args.action in ("pause", "forget", "correct"):
        from .structure_models import RoutingPattern, RoutingReview

        for rule in db.scalars(
            select(RoutingPattern).where(
                RoutingPattern.owner_id == owner, RoutingPattern.origin == "search_candidate"
            )
        ):
            if alias.id in rule.reason:
                rule.status = "paused" if args.action == "pause" else "forgotten"
                rule.revision += 1
        for review in db.scalars(
            select(RoutingReview).where(RoutingReview.owner_id == owner, RoutingReview.status == "pending")
        ):
            review.questions = [q for q in review.questions if q.get("alias_id") != alias.id]
            if not review.questions:
                review.status = "completed"
    alias.revision += 1
    alias.updated_at = now()
    return {"updated": True}


def routing_candidates(db, owner):
    """Propose only. Implicit search evidence never enters the held-out routing dataset."""
    from .structure import ensure
    from .structure_models import RoutingPattern
    from .routing import validate_assignment, classification_question, fingerprint

    schema = ensure(db, owner)
    docs, records = snapshot(db, owner)
    questions = []
    aliases = list(
        db.scalars(
            select(SearchAlias).where(
                SearchAlias.owner_id == owner,
                SearchAlias.account_id == owner,
                SearchAlias.status.in_(["provisional", "confirmed"]),
            )
        )
    )
    for alias in aliases:
        target = docs.get(alias.target_key)
        if not target or target["definition_fingerprint"] != alias.target_fingerprint:
            continue
        accepted = list(
            db.scalars(
                select(SearchSession).where(
                    SearchSession.alias_id == alias.id,
                    SearchSession.outcome == "accepted",
                    SearchSession.suppressed.is_(False),
                )
            )
        )
        if not accepted and alias.status != "confirmed":
            continue
        stamp = digest([alias.target_key, alias.revision, [s.id for s in accepted]])
        if stamp == alias.review_fingerprint:
            continue
        for t in schema.definition["types"]:
            if t.get("archived") or "work" not in t["capabilities"]:
                continue
            assignment = {"values": {}}
            if (
                target["kind"] == "record"
                and target.get("type_id") in t["parent_types"]
                and not set(target.get("capabilities", [])) & {"work", "content"}
            ):
                assignment["parent_id"] = target["record_id"]
            elif target["kind"] == "record":
                fields = [
                    f
                    for f in t["fields"]
                    if not f.get("archived")
                    and f["kind"] == "relation"
                    and target.get("type_id") in f["target_types"]
                ]
                if len(fields) == 1:
                    f = fields[0]
                    assignment["values"][f["id"]] = (
                        [target["record_id"]] if f["multiple"] else target["record_id"]
                    )
            elif target["kind"] == "option" and target.get("type_id") == t["id"]:
                f = next(f for f in t["fields"] if f["id"] == target["field_id"])
                assignment["values"][f["id"]] = (
                    [target["option_id"]] if f["kind"] == "multiselect" else target["option_id"]
                )
            if not assignment.get("parent_id") and not assignment["values"]:
                continue
            try:
                validate_assignment(db, owner, schema, t["id"], assignment)
            except DomainError:
                continue
            condition = {"type_id": t["id"], "phrase": alias.phrase}
            fp = fingerprint({"condition": condition, "assignment": assignment})
            existing = db.scalar(
                select(RoutingPattern).where(
                    RoutingPattern.owner_id == owner, RoutingPattern.fingerprint == fp
                )
            )
            if existing:
                continue
            rule = RoutingPattern(
                owner_id=owner,
                fingerprint=fp,
                condition=condition,
                assignment=assignment,
                evidence_ids=[],
                reason="Suggested from search vocabulary. Search acceptance is not verified organization evidence. Alias "
                + alias.id,
                status="candidate",
                origin="search_candidate",
                schema_revision=schema.revision,
            )
            db.add(rule)
            db.flush()
            questions.append(
                dict(
                    id=rule.id,
                    question=classification_question(db, schema, t["id"], alias.phrase, assignment),
                    evidence_count=0,
                    held_out=0,
                    precision=0,
                    source="search_alias",
                    alias_id=alias.id,
                )
            )
        alias.review_fingerprint = stamp
    return questions
