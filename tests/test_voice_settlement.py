import asyncio

import httpx
from jarvis import budget
from jarvis.db import session_scope
from jarvis.domain import preferences
from jarvis.models import BudgetReservation
from jarvis.voice import Controller


def controller():
    with session_scope() as db:
        c = Controller("davin", "test", "conversation", None, preferences(db, "davin"))
        budget.reserve(db, "davin", c.id, 5, c.model)
    return c


async def test_already_hung_up_idle_call_releases_unused_reservation(monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, *args, **kwargs):
            return httpx.Response(404)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: Client())
    c = controller()
    c.call_id = "ended"
    await c.close(uncertain=True)
    with session_scope() as db:
        assert db.get(BudgetReservation, c.id).state == "closed"


async def test_close_collects_cancelled_response_usage_before_releasing():
    c = controller()
    c.responses["response"] = {"epoch": c.epoch, "phase": "speak", "created": 0}

    async def completed():
        await asyncio.sleep(0.05)
        await c.event(
            {
                "type": "response.done",
                "response": {
                    "id": "response",
                    "status": "cancelled",
                    "usage": {
                        "input_token_details": {"text_tokens": 10},
                        "output_token_details": {"text_tokens": 5},
                    },
                },
            }
        )

    task = asyncio.create_task(completed())
    await c.close()
    await task
    with session_scope() as db:
        row = db.get(BudgetReservation, c.id)
        assert row.state == "closed" and row.actual > 0


async def test_unsettled_generation_keeps_its_reservation():
    c = controller()
    c.responses["missing"] = {"epoch": c.epoch, "phase": "plan", "created": 0}
    c.receiver = asyncio.current_task()
    await c.close()
    with session_scope() as db:
        assert db.get(BudgetReservation, c.id).state == "uncertain"
