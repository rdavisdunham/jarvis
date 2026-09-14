"""Synthetic identity fixture, never mounted in the production application."""
from fastapi.responses import JSONResponse
from jarvis.auth import new_session
from jarvis.db import session_scope
from jarvis.models import GoogleIdentity
from productivity_acceptance import app


@app.post("/api/v1/auth/__test_guest_login")
def guest_login():
    token,csrf=new_session("guest-fixture","google")
    with session_scope() as db:
        if not db.get(GoogleIdentity,"guest-fixture"):
            db.add(GoogleIdentity(owner_id="guest-fixture",subject="guest-fixture-subject",email="guest@example.test"))
    r=JSONResponse({"csrf":csrf})
    r.set_cookie("jarvis_session",token,httponly=True,samesite="strict")
    return r
