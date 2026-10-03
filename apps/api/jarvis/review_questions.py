"""Personal review adapters and durable invitations. Reading/delivery never resolves a question."""

from datetime import timedelta
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from .models import AgentWork, Conversation, Job, MemoryReview, ReviewDelivery, SharedWorkspace, now
from .structure_models import FieldUnderstanding, RoutingPattern, RoutingReview
from .domain import DomainError, advisory, check_revision, emit, owned, preferences, serial
from . import memory_review, routing


def personal(db, owner):
    if db.get(SharedWorkspace, owner):
        raise DomainError(
            "PERSONAL_WORKSPACE", "Questions and learning belong to your personal workspace.", 403
        )


def state(row, status="pending"):
    return "deferred" if status == "pending" and row.deferred_until and row.deferred_until > now() else status


def items(db, owner):
    personal(db, owner)
    advisory(db, "workspace:" + owner)
    advisory(db, "memory:" + owner)
    result = []
    for row in db.scalars(
        select(MemoryReview).where(MemoryReview.owner_id == owner).order_by(MemoryReview.created_at.desc())
    ):
        if row.status == "pending" and not memory_review.review_candidates(db, row):
            row.status = "stale"
            row.revision += 1
            row.resolved_at = now()
        data = memory_review.review_data(db, row)
        status = state(
            row, "pending" if row.status == "pending" else "stale" if row.status == "stale" else "resolved"
        )
        result.append(
            dict(
                key="memory:" + row.id,
                kind="memory",
                revision=row.revision,
                status=status,
                question=data["question"] or "Memory review " + row.status,
                candidates=data["candidates"],
                source_id=row.id,
                deferred_until=row.deferred_until,
                created_at=row.created_at,
                answer_tool="memory.resolve",
                answer_args={"review_id": row.id, "expected_revision": row.revision},
            )
        )
    from .structure import definition_entries, fingerprint

    schema = routing.ensure(db, owner)
    definitions = dict(definition_entries(schema.definition))
    # FieldUnderstanding is canonical; its RoutingReview mirror is deliberately excluded.
    for row in db.scalars(
        select(FieldUnderstanding)
        .where(FieldUnderstanding.owner_id == owner)
        .order_by(FieldUnderstanding.updated_at.desc())
    ):
        if not row.questions:
            continue
        entry = definitions.get(row.definition_id)
        valid = entry and not entry.get("archived") and fingerprint(entry) == row.fingerprint
        result.append(
            dict(
                key="field:" + row.definition_id,
                kind="field",
                revision=row.revision,
                status="stale" if not valid else state(row) if row.status == "needs_input" else "resolved",
                question=row.questions[0],
                source_id=row.definition_id,
                definition_id=row.definition_id,
                deferred_until=row.deferred_until,
                created_at=row.updated_at,
                answer_tool="routing.understand",
                answer_args={"definition_id": row.definition_id, "expected_revision": row.revision},
                evidence=row.understanding,
            )
        )
    schema = None
    for row in db.scalars(
        select(RoutingReview).where(RoutingReview.owner_id == owner).order_by(RoutingReview.created_at.desc())
    ):
        if row.summary.get("definition_id") or row.status == "preview":
            continue
        for q in row.questions:
            rule = db.get(RoutingPattern, q["id"])
            status = state(row) if row.status == "pending" else "resolved"
            if not rule or rule.owner_id != owner or rule.status not in {"candidate", "needs_review"}:
                status = "stale"
            elif status in {"pending", "deferred"}:
                try:
                    schema = schema or routing.ensure(db, owner)
                    routing.validate_assignment(db, owner, schema, rule.condition["type_id"], rule.assignment)
                except DomainError:
                    status = "stale"
            result.append(
                dict(
                    key="routing:" + row.id + ":" + q["id"],
                    kind="routing",
                    revision=row.revision,
                    status=status,
                    question=q["question"],
                    source_id=row.id,
                    deferred_until=row.deferred_until,
                    created_at=row.created_at,
                    evidence={k: v for k, v in q.items() if k != "question"},
                    answer_tool="routing.answer",
                    answer_args={
                        "review_id": row.id,
                        "expected_revision": row.revision,
                        "question_id": q["id"],
                    },
                )
            )
        for n, answer in enumerate(row.answers):
            q = answer.get("question", {})
            result.append(
                dict(
                    key="answered:" + row.id + ":" + str(n),
                    kind="routing",
                    revision=row.revision,
                    status="resolved",
                    question=q.get("question", "Organization review"),
                    source_id=row.id,
                    created_at=row.created_at,
                    evidence=answer,
                    answer_tool=None,
                    answer_args={},
                )
            )
    return result


def learning(db, owner):
    runs = []
    for kind in ("extract_memory", "embed_memory", "review_memory", "review_routing", "assess_field"):
        row = db.scalar(
            select(Job)
            .where(Job.owner_id == owner, Job.kind == kind)
            .order_by(Job.created_at.desc())
            .limit(1)
        )
        result = row.result if row else None
        if row and kind == "review_routing" and row.payload.get("review_id"):
            review = db.get(RoutingReview, row.payload.get("review_id"))
            if review and review.owner_id == owner:
                result = {"review_id": review.id, "summary": review.summary, "job_result": result}
        runs.append(
            {
                "kind": kind,
                "id": row.id if row else None,
                "status": row.status if row else "not_run",
                "created_at": row.created_at if row else None,
                "finished_at": row.finished_at if row else None,
                "result": result,
            }
        )
    return {
        "runs": runs,
        "memory_review": memory_review.status(db, owner),
        "note": "Persisted account jobs and results; a running worker is not evidence of useful learning.",
    }


def listing(db, owner, *, category="all", status="open", offset=0, limit=40):
    rows = items(db, owner)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("pending", "deferred", "resolved", "stale")}
    visible = [
        r
        for r in rows
        if (
            category == "all"
            or (r["kind"] != "memory" if category == "organization" else r["kind"] == category)
        )
        and (status == "all" or r["status"] in ({"pending", "deferred"} if status == "open" else {status}))
    ]
    page = visible[offset : offset + limit]
    for q in page:
        q["agent_tool"] = q["answer_tool"].replace(".", "_", 1) if q.get("answer_tool") else None
        delivery = db.scalar(
            select(ReviewDelivery)
            .where(ReviewDelivery.owner_id == owner, ReviewDelivery.question_key == q["key"])
            .order_by(ReviewDelivery.created_at.desc())
            .limit(1)
        )
        if delivery and delivery.state in {"reserved", "forwarded"} and delivery.expires_at <= now():
            delivery.state = "expired"
            delivery.updated_at = now()
        q["delivery"] = (
            {k: serial(delivery)[k] for k in ("channel", "state", "updated_at")} if delivery else None
        )
    return jsonable_encoder(
        {
            "items": page,
            "counts": counts,
            "total": len(visible),
            "next_offset": offset + limit if offset + limit < len(visible) else None,
            "learning": learning(db, owner),
        }
    )


def find(db, owner, key, revision):
    row = next((r for r in items(db, owner) if r["key"] == key), None)
    if not row or row["revision"] != revision or row["status"] not in {"pending", "deferred"}:
        raise DomainError("STALE_REVIEW", "This question changed. Refresh Questions before answering.", 409)
    return row


def defer(db, owner, args):
    advisory(db, "workspace:" + owner)
    advisory(db, "review-delivery:" + owner)
    q = find(db, owner, args.question_key, args.expected_revision)
    model = (
        MemoryReview
        if q["kind"] == "memory"
        else RoutingReview
        if q["kind"] == "routing"
        else FieldUnderstanding
    )
    row = (
        db.get(model, (owner, q["source_id"]))
        if model is FieldUnderstanding
        else owned(db, model, q["source_id"], owner, lock=True)
    )
    check_revision(row, args.expected_revision)
    row.deferred_until = (
        now()
        + (
            {"today": timedelta(hours=3), "tomorrow": timedelta(days=1), "week": timedelta(days=7)}[
                args.until
            ]
        )
    )
    row.revision += 1
    emit(
        db,
        owner,
        "routing.changed" if q["kind"] != "memory" else "memory.changed",
        q["source_id"],
        row.revision,
    )
    return {
        "question_key": q["key"],
        "status": "deferred",
        "revision": row.revision,
        "deferred_until": row.deferred_until.isoformat(),
    }


def work_pending(db, owner, conversation_id):
    return (
        db.scalar(
            select(AgentWork.id)
            .join(Job, Job.id == AgentWork.id)
            .where(
                AgentWork.owner_id == owner,
                AgentWork.conversation_id == conversation_id,
                AgentWork.result["archived_at"].as_string().is_(None),
                Job.status.in_(
                    ["queued", "dispatched", "running", "retrying", "waiting_sync", "needs_input"]
                ),
            )
            .limit(1)
        )
        is not None
    )


def reserve(db, owner, device, conversation_id, channel):
    personal(db, owner)
    conversation = owned(db, Conversation, conversation_id, owner)
    if conversation.device_id != device:
        raise DomainError("NOT_AUTHORIZED", "Open this conversation on its original device.", 403)
    advisory(db, "workspace:" + owner)
    advisory(db, "review-delivery:" + owner)
    attempts = list(
        db.scalars(
            select(ReviewDelivery)
            .where(ReviewDelivery.owner_id == owner)
            .order_by(ReviewDelivery.created_at.desc())
        )
    )
    for attempt in attempts:
        if attempt.state in {"reserved", "forwarded"} and attempt.expires_at <= now():
            attempt.state = "expired"
            attempt.updated_at = now()
    if work_pending(db, owner, conversation_id):
        return None
    if any(a.state in {"reserved", "forwarded"} and a.expires_at > now() for a in attempts):
        return None
    # At most one invitation per conversation/day; visible text also throttles other devices.
    if any(
        a.conversation_id == conversation_id
        and a.created_at > now() - timedelta(days=1)
        or a.state == "presented"
        and a.updated_at > now() - timedelta(days=1)
        or a.created_at > now() - timedelta(minutes=10)
        for a in attempts
    ):
        return None
    prefs = preferences(db, owner)
    eligible = [
        r
        for r in items(db, owner)
        if r["status"] == "pending"
        and (
            prefs["deep_sleep_enabled"] and prefs["memory_learning"]
            if r["kind"] == "memory"
            else prefs["routing_review_enabled"]
        )
    ]
    if not eligible:
        return None
    q = eligible[0]
    row = ReviewDelivery(
        owner_id=owner,
        device_id=device,
        conversation_id=conversation_id,
        question_key=q["key"],
        question_revision=q["revision"],
        channel=channel,
        expires_at=now() + timedelta(minutes=2),
    )
    db.add(row)
    db.flush()
    label = "a memory detail" if q["kind"] == "memory" else "your organization"
    return {
        "id": row.id,
        "question_key": q["key"],
        "question_revision": q["revision"],
        "message": "I have a question about " + label + " when you are ready. Open Questions to review it.",
        "expires_at": row.expires_at.isoformat(),
        "state": row.state,
    }


def acknowledge(db, owner, device, identity, event):
    advisory(db, "workspace:" + owner)
    advisory(db, "review-delivery:" + owner)
    row = owned(db, ReviewDelivery, identity, owner, lock=True)
    if row.device_id != device:
        raise DomainError("NOT_AUTHORIZED", "This invitation belongs to another device.", 403)
    if row.state in {"presented", "interrupted", "expired"}:
        return serial(row)
    if row.expires_at <= now():
        event = "expired"
    elif event != "interrupted":
        try:
            find(db, owner, row.question_key, row.question_revision)
        except DomainError:
            event = "expired"
        if work_pending(db, owner, row.conversation_id):
            event = "interrupted"
    if event == "presented" and row.channel != "text":
        raise DomainError("INVALID_ARGUMENT", "Voice forwarding is not proof of completed playback.")
    row.state = event
    row.updated_at = now()
    return serial(row)
