"""Isolated site-control fixture; never mounted in production."""

from uuid import uuid4

from jarvis.api import User, app
from jarvis.tools import call_tool


@app.post("/api/v1/__test_ui")
async def ui_fixture(body: dict, user: User):
    return await call_tool(
        user.owner_id, str(uuid4()), 0, body["name"], body["arguments"], device=user.device_id
    )
