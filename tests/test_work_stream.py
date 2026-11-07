"""Dani's reply streams to the asking device as it is written (work_stream.py)."""

import asyncio
import json

import psycopg
from jarvis import work_stream
from jarvis.agent_link import listen_dsn


def test_batches_reach_listeners_through_postgres_notify():
    with psycopg.connect(listen_dsn(), autocommit=True) as listener:
        listener.execute(f"LISTEN {work_stream.CHANNEL}")
        deltas = work_stream.Deltas("davin", "device-1", "work-1")
        deltas.start_round(10)
        for piece in ("Added ", "Alpha", "."):
            deltas.add(piece)
        deltas.flush()
        deltas.add("é" * (work_stream.MAX_CHARS + 5))  # Long text splits under NOTIFY's limit.
        deltas.flush()
        notes = [json.loads(n.payload) for n in listener.notifies(timeout=2, stop_after=3)]
    text = "".join(n["t"] for n in notes)
    assert text == "Added Alpha." + "é" * (work_stream.MAX_CHARS + 5)
    assert {(n["o"], n["d"], n["w"], n["r"]) for n in notes} == {("davin", "device-1", "work-1", 10)}
    assert [n["s"] for n in notes] == sorted(n["s"] for n in notes)


def test_only_the_asking_owner_and_device_receive_a_batch():
    async def scenario():
        mine, other_device, other_owner = (work_stream.subscribe("davin", "d1"), work_stream.subscribe("davin", "d2"),
                                           work_stream.subscribe("someone", "d1"))
        try:
            work_stream.deliver(json.dumps({"o": "davin", "d": "d1", "w": "w", "r": 0, "s": 1, "t": "Hi"}))
            work_stream.deliver("not json")
            got = await work_stream.drain(mine, 0.2)
            assert got == [{"w": "w", "r": 0, "s": 1, "t": "Hi"}]
            assert other_device.empty() and other_owner.empty()
            assert await work_stream.drain(mine, 0.05) == []
        finally:
            for owner, device, queue in (("davin", "d1", mine), ("davin", "d2", other_device),
                                         ("someone", "d1", other_owner)):
                work_stream.unsubscribe(owner, device, queue)
        assert not work_stream._subscribers
    asyncio.run(scenario())


def test_a_broken_database_stops_the_preview_not_the_reply(monkeypatch):
    from jarvis import db

    def broken():
        raise RuntimeError("database down")
    monkeypatch.setattr(db, "engine", broken)
    deltas = work_stream.Deltas("davin", "d1", "w")
    deltas.add("text")
    deltas.flush()
    assert deltas.failed
    deltas.add("more")  # Ignored from now on.
    deltas.flush()
