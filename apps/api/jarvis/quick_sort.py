"""One-at-a-time Home filing with server-bound, explicitly human evidence."""
import copy
import re
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from .access import actor, authorize_execution, execution
from .domain import DomainError, advisory, owned
from .models import AgentWork, Command, Job, Note, Task, now
from .structure_models import QuickSortSession, StructureRecord
from .text_normalize import canon
from .work_crypto import unseal

CONTROL_WORDS = {
    "skip": {"skip", "skip it", "skip this", "skip that", "skip this one", "next", "next one", "leave unfiled", "leave it unfiled"},
    "pause": {"pause", "pause sorting", "pause quick sort", "pause for now"},
    "resume": {"resume", "resume sorting", "resume quick sort", "continue sorting", "continue quick sort"},
    "stop": {"stop", "stop sorting", "stop quick sort", "done sorting", "finish sorting"},
}
YES = {"yes", "yes please", "yep", "yeah", "correct", "that is correct", "thats correct", "confirm", "confirmed"}
NO = {"no", "nope", "neither", "none of those", "not that one"}
UNRELATED = re.compile(r"^(?:please )?(?:add|create|delete|remind|schedule|show|find|search|open|rename|complete|what|when|why|how)\b|\b(?:and|also|then) (?:add|create|delete|remind|schedule|show|find|search|open|rename|complete)\b")
NEGATION = re.compile(r"\b(?:not|dont|don t|never|except|neither)\b")
ORDINALS = {"first": 0, "first one": 0, "the first one": 0, "second": 1, "second one": 1,
            "the second one": 1, "third": 2, "third one": 2, "the third one": 2}


def clean(text):
    return canon(text).strip()


def control_word(text):
    value = re.sub(r"^(?:lets|let s) ", "", clean(text))
    return next((name for name, words in CONTROL_WORDS.items() if value in words), None)


def destination(text, *, allow_conjunction=False):
    """Only a named choice, not any sentence mentioning a Home."""
    value = clean(text)
    if not value or UNRELATED.search(value) or NEGATION.search(value):
        return None
    value = re.sub(r"^(?:okay |ok |please )", "", value)
    value = re.sub(r" please$", "", value)
    value = re.sub(r"^(?:put|file|move|place|assign|sort) (?:it|this|that|the task|this task|that task|the note|this note|that note|this one|that one) (?:in|into|under|to) ", "", value)
    value = re.sub(r"^(?:it|this|that) (?:belongs|goes) (?:in|under|to) ", "", value)
    value = re.sub(r"^(?:in|under|to) ", "", value)
    if not allow_conjunction and re.search(r"\b(?:and|or)\b", value):
        return None
    return value


def is_answer(text):
    value = clean(text)
    return bool(value) and len(value) <= 600 and not UNRELATED.search(value) and value not in {
        "goodbye", "bye", "thank you", "thanks", "end voice"}


def current_work(db, owner):
    from .bot_access import current_id
    authorize_execution(db, owner, write=True)
    grant = execution.get()
    work = db.get(AgentWork, grant["id"]) if grant else None
    if not work or work.owner_id != owner or work.credential_id or current_id():
        raise DomainError("HUMAN_CONVERSATION_REQUIRED", "Use Quick Sort in your own text or voice conversation.", 403)
    return work


def require_session(db, owner, identity, *, lock=False):
    from .bot_access import current_id
    if current_id():
        raise DomainError("NOT_AUTHORIZED", "Quick Sort belongs to a human conversation.", 403)
    row = owned(db, QuickSortSession, str(identity), owner, lock=lock)
    if row.account_id != actor(db, owner):
        raise DomainError("NOT_FOUND", "That sorting session is not available.", 404)
    return row


def save(row, data):
    row.state = copy.deepcopy(data)
    flag_modified(row, "state")
    row.revision += 1
    row.updated_at = now()


def compact(db, row, schema):
    from .structure import data
    value = data(db, row, schema)
    result = {key: value.get(key) for key in
              ("id", "title", "type_id", "type_name", "revision", "parent_id", "source")}
    result["home"] = " / ".join(x["title"] for x in value.get("home", []))[-600:]
    return result


def leaf(db, row, kind="both"):
    if row.archived or db.scalar(select(StructureRecord.id).where(
            StructureRecord.owner_id == row.owner_id, StructureRecord.parent_id == row.id,
            StructureRecord.archived.is_(False)).limit(1)):
        return False
    if row.task_id and kind in {"tasks", "both"}:
        task = db.get(Task, row.task_id)
        return bool(task and not task.archived and not task.is_template and not task.is_quick_list
                    and task.status not in {"completed", "cancelled"})
    if row.note_id and kind in {"notes", "both"}:
        note = db.get(Note, row.note_id)
        return bool(note and not note.archived)
    return False


def homes(db, owner, item, schema):
    from .structure import record_type, validate_parent, parent_chain, fingerprint
    kind = record_type(schema, item.type_id)
    result = []
    for row in db.scalars(select(StructureRecord).where(StructureRecord.owner_id == owner,
            StructureRecord.archived.is_(False), StructureRecord.type_id.in_(kind["parent_types"]))
            .order_by(StructureRecord.title, StructureRecord.id)):
        try:
            validate_parent(db, owner, item, kind, row.id)
        except DomainError:
            continue
        chain = [*reversed(parent_chain(db, owner, row.parent_id)), row]
        if any(parent.archived for parent in chain):
            continue
        names = [parent.title for parent in chain]
        result.append({"id": row.id, "title": row.title, "path": " / ".join(names)[-600:],
                       "match_path": clean(" ".join(names)),
                       "fingerprint": fingerprint([(p.id, p.revision, p.title, p.parent_id) for p in chain])})
    return result


def public_home(home):
    return {key: home[key] for key in ("id", "title", "path")}


def item_for(db, row):
    data = row.state
    identity = data["items"][data["cursor"]] if data["cursor"] < len(data["items"]) else None
    item = db.get(StructureRecord, identity) if identity else None
    return item if item and item.owner_id == row.owner_id else None


def public(db, row):
    from .structure import ensure
    data, item = row.state, item_for(db, row)
    return {"workflow": "quick_sort", "session_id": row.id, "status": row.status, "revision": row.revision,
            "progress": {"total": len(data["items"]), "reviewed": data["cursor"],
                         "filed": len(data["filed"]), "skipped": len(data["skipped"])},
            "current": compact(db, item, ensure(db, row.owner_id)) if item and not item.archived else None,
            "candidates": [public_home(home) for home in data.get("candidates", [])],
            "question": data.get("question") if row.status == "active" else None,
            "message": data.get("message", ""),
            "pending_request_id": data.get("pending_work_id"), "clarification_id": data.get("question_id"),
            "learning": "Only your explicit Home choice is evidence; suggestions and skipped items are not."}


def state(db, owner, session_id=None):
    from .bot_access import current_id
    authorize_execution(db, owner)
    if current_id():
        raise DomainError("NOT_AUTHORIZED", "Quick Sort belongs to a human conversation.", 403)
    if session_id:
        return public(db, require_session(db, owner, session_id))
    rows = db.scalars(select(QuickSortSession).where(QuickSortSession.owner_id == owner,
            QuickSortSession.account_id == actor(db, owner)).order_by(QuickSortSession.updated_at.desc()).limit(10))
    return {"sessions": [public(db, row) for row in rows]}


def ask(db, row, work, *, prefix="", candidates=None, proposal=None):
    from .structure import ensure, reconcile_core
    data = copy.deepcopy(row.state)
    while data["cursor"] < len(data["items"]):
        item = db.get(StructureRecord, data["items"][data["cursor"]])
        if item and item.owner_id == row.owner_id and leaf(db, item):
            break
        data["skipped"].append(data["items"][data["cursor"]])
        data["cursor"] += 1
    row.state = data
    item = item_for(db, row)
    if item is None:
        row.status = "completed"
        data.update(question=None, pending_work_id=None, question_id=None, proposal=None, candidates=[],
                    message=(prefix + f" Quick Sort finished: {len(data['filed'])} filed and {len(data['skipped'])} skipped.").strip())
        save(row, data)
        return public(db, row)
    schema = ensure(db, row.owner_id)
    reconcile_core(db, item, schema)
    selected = (candidates or [])[:5]
    if proposal:
        question = f'File "{item.title[:200]}" under {proposal["path"]}?'
    elif selected:
        choices = "; ".join(f'{i+1}: {h["path"][:150]}' for i, h in enumerate(selected))
        question = f'For "{item.title[:160]}", which Home do you mean? {choices}. Say a number or a different Home.'
    else:
        question = f'Where should "{item.title[:300]}" live? Say its Home, or say skip, pause, or stop.'
    data.update(item_revision=item.revision, schema_revision=schema.revision,
                question=(prefix + " " + question).strip(), pending_work_id=None, question_id=None,
                awaiting_work_id=work.id, awaiting_revision=work.revision,
                candidates=selected, proposal=proposal, message=prefix)
    row.conversation_id, row.status = work.conversation_id, "active"
    save(row, data)
    return public(db, row)


def start(db, owner, args, command_id, work):
    from .structure import ensure, core_drift, sync_core_records
    from .hierarchy import adopt
    from .record_contents import subtree_query
    schema = ensure(db, owner)
    if core_drift(db, owner):
        sync_core_records(db, owner, schema)
    adopt(db, owner, schema)
    active = db.scalar(select(QuickSortSession).where(QuickSortSession.owner_id == owner,
            QuickSortSession.account_id == work.account_id, QuickSortSession.status == "active",
            QuickSortSession.conversation_id == work.conversation_id).order_by(QuickSortSession.created_at.desc()))
    if active:
        raise DomainError("SORT_ALREADY_ACTIVE", "Read quicksort_state, then answer, pause, or stop the active sorting session.",
                          409, {"session_id": active.id})
    q = select(StructureRecord).where(StructureRecord.owner_id == owner, StructureRecord.archived.is_(False))
    if args.scope == "unfiled":
        q = q.where(StructureRecord.parent_id.is_(None))
    elif args.scope == "home":
        owned(db, StructureRecord, str(args.home_id), owner)
        q = q.where(StructureRecord.id.in_(subtree_query(owner, str(args.home_id))))
    else:
        for identity in args.record_ids:
            owned(db, StructureRecord, str(identity), owner)
        q = q.where(StructureRecord.id.in_([str(x) for x in args.record_ids]))
    items = []
    for item in db.scalars(q.order_by(StructureRecord.created_at, StructureRecord.id)):
        if leaf(db, item, args.kind):
            items.append(item.id)
        if len(items) == args.limit:
            break
    row = QuickSortSession(owner_id=owner, account_id=work.account_id, conversation_id=work.conversation_id,
        state={"items": items, "cursor": 0, "filed": [], "skipped": [], "consumed": []})
    db.add(row)
    db.flush()
    return ask(db, row, work)


def pending_session(db, work):
    return db.scalar(select(QuickSortSession).where(QuickSortSession.owner_id == work.owner_id,
        QuickSortSession.account_id == work.account_id, QuickSortSession.status == "active",
        QuickSortSession.state["pending_work_id"].as_string() == work.id))


def before_answer(db, current, target):
    row = pending_session(db, target)
    if row and (current.conversation_id != row.conversation_id
                or not is_answer(unseal(current.input_ciphertext).get("message", ""))):
        raise DomainError("UNRELATED_SORT_ANSWER", "Handle this separate request independently and leave Quick Sort waiting.", 409)


def capture_answer(db, current, target, question, data):
    row = pending_session(db, target)
    if row:
        data["quicksort_answer"] = {"session_id": row.id, "question_id": question["id"],
                                   "answer": data["message"], "revision": current.revision + 1}
        saved = copy.deepcopy(row.state)
        saved.update(awaiting_work_id=current.id, awaiting_revision=current.revision + 1)
        save(row, saved)


def capture_revision(db, work, message, data):
    row = pending_session(db, work)
    if not row:
        return
    if not message or not message.strip():
        raise DomainError("SORT_ANSWER_REQUIRED", "Enter a Home choice or a Quick Sort control before continuing.", 409)
    if not is_answer(message):
        raise DomainError("UNRELATED_SORT_ANSWER", "Send that as a separate chat request; this action is waiting for a Home choice.", 409)
    data["quicksort_answer"] = {"session_id": row.id, "question_id": row.state["question_id"],
                               "answer": message, "revision": work.revision + 1}
    saved = copy.deepcopy(row.state)
    saved.update(awaiting_work_id=work.id, awaiting_revision=work.revision + 1)
    save(row, saved)


def answer_text(row, work):
    captured = unseal(work.input_ciphertext).get("quicksort_answer", {})
    if (row.status != "active" or work.conversation_id != row.conversation_id
            or captured.get("session_id") != row.id or captured.get("question_id") != row.state.get("question_id")
            or not row.state.get("question_id") or captured.get("revision") != work.revision):
        raise DomainError("SORT_ANSWER_REQUIRED", "Answer the pending question through work_answer before filing; quicksort_state has its exact IDs.", 409)
    key = f'{work.id}:{work.revision}:{captured["question_id"]}'
    if key in row.state.get("consumed", []):
        raise DomainError("SORT_ANSWER_USED", "That answer already handled one item. Wait for a new answer.", 409)
    return captured["answer"], key, captured["question_id"]


def reply(db, owner, args, command_id, work):
    from .structure import ensure, reconcile_core
    row = require_session(db, owner, args.session_id, lock=True)
    text, key, question_id = answer_text(row, work)
    data = copy.deepcopy(row.state)
    data["consumed"].append(key)
    save(row, data)
    action = control_word(text)
    if action in {"skip", "pause", "stop"}:
        return control(db, owner, args.session_id, action, work, trusted=True)
    item, schema = item_for(db, row), ensure(db, owner)
    if not item or not leaf(db, item):
        return ask(db, row, work, prefix="That item is no longer available to file.")
    reconcile_core(db, item, schema)
    if item.revision != row.state["item_revision"] or schema.revision != row.state["schema_revision"]:
        return ask(db, row, work, prefix="The item or structure changed. Please review it again.")
    available, proposed, chosen = homes(db, owner, item, schema), row.state.get("proposal"), None
    value = clean(text)
    ordinals = {clean(label): index for label, index in ORDINALS.items()}
    if proposed and value in YES:
        chosen = next((h for h in available if h["id"] == proposed["id"] and h["fingerprint"] == proposed["fingerprint"]), None)
        if not chosen:
            return ask(db, row, work, prefix="That proposed Home changed. Please choose again.")
    elif value in NO:
        return ask(db, row, work)
    elif row.state.get("candidates") and (value in ordinals or value.isdecimal()):
        index = ordinals.get(value, int(value)-1 if value.isdecimal() else -1)
        candidates = row.state["candidates"]
        proposal = candidates[index] if 0 <= index < len(candidates) else None
        chosen = next((h for h in available if proposal and h["id"] == proposal["id"] and h["fingerprint"] == proposal["fingerprint"]), None)
        if not chosen:
            return ask(db, row, work, prefix="Choose one of the current Homes.")
    else:
        phrase = destination(text)
        exact_phrase = destination(text, allow_conjunction=True)
        exact = [h for h in available if exact_phrase and exact_phrase in {clean(h["title"]), h["match_path"]}]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(exact) > 1:
            return ask(db, row, work, candidates=exact)
        elif args.home_id and phrase:
            proposal = next((h for h in available if h["id"] == str(args.home_id)), None)
            if not proposal:
                raise DomainError("INVALID_PARENT", "That is not a legal current Home.")
            return ask(db, row, work, proposal=proposal)
        else:
            return ask(db, row, work, prefix="I need one specific Home; I have not changed this item.")
    from .domain import execute
    values = {"record_id": item.id, "expected_revision": item.revision, "schema_revision": schema.revision,
              "parent_id": chosen["id"]}
    if item.parent_id == chosen["id"]:
        values["sort_order"] = item.sort_order
    receipt_id = command_id + ":home"
    execute(db, owner, receipt_id, "record.update", values)
    from .routing import observe_home_choice
    observe_home_choice(db, item, receipt_id, {"kind": "quick_sort", "session_id": row.id,
        "work_id": work.id, "question_id": question_id, "answer_revision": work.revision})
    data = copy.deepcopy(row.state)
    data["filed"].append({"record_id": item.id, "parent_id": chosen["id"], "command_id": receipt_id})
    data["cursor"] += 1
    save(row, data)
    result = ask(db, row, work, prefix=f'Filed "{item.title[:160]}" under {chosen["path"][:300]}.')
    result["saved_command_id"] = receipt_id
    return result


def retire_question(db, row, current):
    from .agent_work import cancel
    identity = row.state.get("pending_work_id")
    pending = db.get(AgentWork, identity) if identity else None
    if pending and pending.id != current.id and db.get(Job, pending.id).status == "needs_input":
        # Retire the old clarification without treating this workflow control as work failure.
        saved_status, saved_state = row.status, copy.deepcopy(row.state)
        cancel(db, pending)
        row.status, row.state = saved_status, saved_state


def control(db, owner, identity, action, work, *, trusted=False):
    row = require_session(db, owner, identity, lock=True)
    if not trusted:
        data = unseal(work.input_ciphertext)
        spoken = data.get("corrections", [])[-1] if data.get("corrections") else data.get("message", "")
        if control_word(spoken) != action:
            raise DomainError("SORT_CONTROL_REQUIRED", "Ask explicitly to skip, pause, resume, or stop Quick Sort.", 409)
    if action != "resume" and work.conversation_id != row.conversation_id:
        raise DomainError("SORT_CONVERSATION_CHANGED", "Resume Quick Sort in this conversation first.", 409)
    data = copy.deepcopy(row.state)
    key = f"control:{work.id}:{work.revision}"
    if key in data["consumed"]:
        raise DomainError("SORT_ANSWER_USED", "That turn already controlled Quick Sort. Wait for a new turn.", 409)
    data["consumed"].append(key)
    save(row, data)
    retire_question(db, row, work)
    if action == "resume":
        if row.status not in {"paused", "active"}:
            raise DomainError("SORT_FINISHED", "Start a new sorting session to review more items.")
        return ask(db, row, work, prefix="Resuming Quick Sort.")
    data = copy.deepcopy(row.state)
    if action == "skip":
        if row.status != "active":
            raise DomainError("SORT_PAUSED", "Resume Quick Sort before skipping an item.")
        if data["cursor"] < len(data["items"]):
            data["skipped"].append(data["items"][data["cursor"]])
            data["cursor"] += 1
        save(row, data)
        return ask(db, row, work, prefix="Skipped.")
    row.status = "paused" if action == "pause" else "stopped"
    data.update(question=None, pending_work_id=None, question_id=None, awaiting_work_id=None,
                message="Quick Sort paused; saved choices remain." if action == "pause" else "Quick Sort stopped; saved choices remain.")
    save(row, data)
    return public(db, row)


def mutate(db, owner, tool, args, command_id):
    work = current_work(db, owner)
    advisory(db, "quick-sort:" + owner + ":" + work.account_id)
    if tool == "quicksort.start":
        return start(db, owner, args, command_id, work)
    if tool == "quicksort.reply":
        return reply(db, owner, args, command_id, work)
    return control(db, owner, args.session_id, args.action, work)


def runner_question(owner, work_id, name, outcome):
    """Read a server-persisted question, never a model-provided pause flag."""
    if name not in {"quicksort_start", "quicksort_reply", "quicksort_control"} or not outcome.get("command_id"):
        return None
    from .db import session_scope
    with session_scope() as db:
        receipt = db.get(Command, (owner, outcome["command_id"]))
        data = receipt.result.get("data", {}) if receipt else {}
        row = db.get(QuickSortSession, data.get("session_id")) if data.get("workflow") == "quick_sort" else None
        work = db.get(AgentWork, work_id)
        if (row and work and row.owner_id == owner and row.account_id == work.account_id
                and row.status == "active" and row.state.get("awaiting_work_id") == work_id
                and row.state.get("awaiting_revision") == work.revision
                and row.state.get("question") == data.get("question")):
            return row.state["question"]
    return None


def finished(db, work, status, message, result):
    rows = list(db.scalars(select(QuickSortSession).where(QuickSortSession.owner_id == work.owner_id,
        QuickSortSession.account_id == work.account_id, QuickSortSession.status == "active",
        QuickSortSession.state["awaiting_work_id"].as_string() == work.id)))
    for row in rows:
        data = copy.deepcopy(row.state)
        if status == "needs_input" and data.get("question") == message and result.get("clarification"):
            data.update(pending_work_id=work.id, question_id=result["clarification"]["id"])
        elif status in {"cancelled", "expired", "failed", "partial", "succeeded", "needs_input"}:
            row.status = "paused"
            data.update(pending_work_id=work.id if status == "needs_input" else None,
                        question_id=(result.get("clarification") or {}).get("id") if status == "needs_input" else None,
                        question=None, message="Quick Sort paused; say resume quick sort to continue. Saved choices remain.")
        save(row, data)
