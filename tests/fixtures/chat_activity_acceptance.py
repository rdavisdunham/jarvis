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
