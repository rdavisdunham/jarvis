"""Synthetic queue lifecycle for browser acceptance; never mounted in production."""
from unittest.mock import patch
from jarvis.api import User
from jarvis import agent_work
from jarvis.db import session_scope
from jarvis.domain import capture_source, execute
from jarvis.models import Conversation
from accounts_acceptance import app


@app.post("/api/v1/__test_work")
def accept_work(body: dict, user: User):
    with session_scope() as db:
        account = agent_work.account_for(db, user.owner_id, user.device_id)
        # External services stay disabled. Only model selection is stubbed; queue,
        # commands, receipts, history and authentication use the real implementation.
        with patch("jarvis.agent_models.selected", return_value=agent_work.agent_models.catalog()["luna"]):
            row = agent_work.enqueue(db, user.owner_id, account, user.device_id,
                body["conversation_id"], body["turn_id"], body["message"])
        capture_source(db, user.owner_id, body["message"], "work:"+row.id+":user",
            role="user", conversation=db.get(Conversation,row.conversation_id))
        return agent_work.public(db,row)


@app.post("/api/v1/__test_work/{request_id}/finish")
def finish_work(request_id: str, body: dict, user: User):
    with session_scope() as db:
        row = agent_work.require_work(db,user.owner_id,user.account_id,request_id)
        navigation = body.get("navigation", False)
        status = body.get("status", "succeeded")
        if not navigation and status == "succeeded":
            with agent_work.principal_for(row):
                execute(db,row.owner_id,row.id+":0","task.create",{"title":body.get("title","Synthetic task")})
        message = "Here is your calendar." if navigation else "Saved the task." if status=="succeeded" else "Which day?"
        agent_work.finish(db,row,status,message,navigation_only=navigation)
        capture_source(db,row.owner_id,message,f"work:{row.id}:assistant:{row.revision}",
            role="assistant",conversation=db.get(Conversation,row.conversation_id))
        return agent_work.public(db,row)


@app.post("/api/v1/__test_search_index")
def index_search(user: User):
    from jarvis import search_index, search_service
    from jarvis.config import get_settings
    def synthetic_vectors(owner,texts,*args):
        return [[1.,0.,0.] if any(w in t.lower() for w in ('pest','termite','insect','extermin','spraying')) else [0.,1.,0.] if 'software' in t.lower() else [0.,0.,1.] for t in texts]
    get_settings().semantic_search_enabled=True
    search_index.embeddings=search_service.embeddings=synthetic_vectors
    with session_scope() as db:identity=search_index.queue_index(db,user.owner_id,force=True)
    search_index.index_workspace(identity)
    return {'indexed':True}


@app.post("/api/v1/__test_organize_note")
def organize_note(body: dict, user: User):
    from sqlalchemy import select
    from jarvis.domain import owned
    from jarvis.models import Job, Note
    from jarvis.note_lists import process, active_lists
    from jarvis.note_list_schema import OrganizationResult
    with session_scope() as db:
        note = owned(db, Note, body["note_id"], user.owner_id)
        source = note.content
        movie = next(r for r in active_lists(db, user.owner_id) if "movies" in r.filters.get("tags", []))
        jid = db.scalar(select(Job.id).where(Job.owner_id == user.owner_id, Job.kind == "organize_note",
            Job.payload["note_id"].as_string() == note.id).order_by(Job.created_at.desc()).limit(1))
    result = OrganizationResult(classifications=[], entries=[
        {"title": title, "evidence": source, "list_ids": [movie.id], "confidence": .99,
         "save_intent": True, "existing_note_id": None} for title in ("Arrival", "Dune")])
    with patch("jarvis.routing.infer", return_value=result):
        process(jid)
    return {"processed": True}

@app.post('/api/v1/__test_voice_draft')
def create_voice_draft(body: dict, user: User):
    from uuid import uuid4
    from jarvis.work_intake import open_voice, append_voice, claim_voice
    with session_scope() as db:
        row = open_voice(db, str(uuid4()), user.owner_id, user.account_id, user.device_id, body['conversation_id'])
        append_voice(db, row.id, 'synthetic', 'user', body['message'], 0, 1000)
        claim_voice(db, row, close=True)
        return {'id': row.id}


@app.post('/api/v1/__test_voice_draft/{identity}/send')
def send_voice_draft(identity: str, body: dict, user: User):
    from jarvis.work_intake import resolve_draft
    with patch('jarvis.config.require_external_services'), patch('jarvis.agent_models.selected', return_value=agent_work.agent_models.catalog()['luna']), session_scope() as db:
        return resolve_draft(db, user.owner_id, user.account_id, user.device_id, identity, message=body['message'])


@app.post("/api/v1/__test_reviews")
def seed_reviews(user: User):
    from jarvis import structure
    from jarvis.models import Memory,MemoryReview,Job,now
    from jarvis.structure_models import FieldUnderstanding
    from uuid import uuid4
    with session_scope() as db:
        structure.ensure(db,user.owner_id)
        field=db.get(FieldUnderstanding,(user.owner_id,"type:client"))
        field.status="needs_input";field.questions=["Does Client mean a company or a person?"];field.revision+=1
        memories=[]
        for name in ("Hayes","Haze"):
            source=capture_source(db,user.owner_id,"My cat is "+name,str(uuid4()),explicit=True)
            m=Memory(owner_id=user.owner_id,source_id=source.id,content=source.content,fingerprint=str(uuid4()))
            db.add(m);db.flush();memories.append(m)
        review=MemoryReview(owner_id=user.owner_id,pair_key=str(uuid4()),memory_ids=[m.id for m in memories],memory_revisions=[m.revision for m in memories])
        db.add(review)
        db.add(Job(owner_id=user.owner_id,kind="review_memory",payload={},status="succeeded",finished_at=now(),result={"scanned":2,"merged":0,"queued_questions":1}))
        return {"seeded":True}
