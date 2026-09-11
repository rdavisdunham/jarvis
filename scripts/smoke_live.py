"""Opt-in integration acceptance against this owner's deployment; creates labeled synthetic records."""
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone, timedelta
import time
import httpx
from dotenv import dotenv_values

root=Path(__file__).resolve().parents[1]
config=dotenv_values(root/".env.upgrade")
with httpx.Client(base_url=config["JARVIS_ORIGIN"],timeout=70) as client:
    login=client.post("/api/v1/auth/login",json={"token":config["JARVIS_OWNER_TOKEN"]})
    login.raise_for_status()
    client.headers["X-CSRF-Token"]=login.json()["csrf"]
    def command(tool,arguments):
        result=client.post("/api/v1/commands",json={"command_id":str(uuid4()),"tool":tool,"arguments":arguments})
        result.raise_for_status()
        return result.json()["data"]
    conv=client.post("/api/v1/conversations",json={"private":True}).json()
    turn=str(uuid4())
    title="Acceptance check — model tool "+turn[:8]
    body={"turn_id":turn,"conversation_id":conv["id"],"message":"Create exactly one task titled '"+title+"' with no due date. This is a synthetic integration test."}
    started=time.monotonic()
    answer=client.post("/api/v1/chat",json=body)
    answer.raise_for_status()
    latency=time.monotonic()-started
    result=answer.json()
    tasks=client.get("/api/v1/tasks").json()["items"]
    matches=[t for t in tasks if t["title"]==title]
    evidence={"provider_status":result["status"],"actions":result["actions"],"model_task_count":len(matches),"chat_seconds":round(latency,2)}
    assert result["status"]=="succeeded" and len(matches)==1, evidence
    repeat=client.post("/api/v1/chat",json=body)
    repeat.raise_for_status()
    assert len([t for t in client.get("/api/v1/tasks").json()["items"] if t["title"]==title])==1
    assert client.get("/api/v1/conversations/"+conv["id"]).json()["messages"]==[]
    schedule=command("schedule.create",{"title":"Acceptance check — durable reminder","when":(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),"timezone":"America/Chicago"})
    notices=[]
    for _ in range(35):
        notices=client.get("/api/v1/notifications").json()["items"]
        if any(n["title"]==schedule["title"] for n in notices): break
        time.sleep(1)
    notice=next(n for n in notices if n["title"]==schedule["title"])
    memory=command("memory.capture",{"content":"Acceptance check: I prefer cedar notebooks."})
    assert client.get("/api/v1/memory",params={"q":"cedar"}).json()["items"]
    corrected=command("memory.correct",{"memory_id":memory["id"],"content":"Acceptance check: I prefer paper notebooks."})
    assert not client.get("/api/v1/memory",params={"q":"cedar"}).json()["items"]
    command("memory.forget",{"memory_id":corrected["id"],"delete_source":True})
    command("memory.forget",{"memory_id":memory["id"],"delete_source":True})
    evidence.update({"task_id":matches[0]["id"],"schedule_id":schedule["id"],"notification_id":notice["id"],"private_transcripts":0,"memory_lifecycle":"passed","duplicate_turn":"passed","budget":client.get("/api/v1/bootstrap").json()["budget"]})
    (root/".runtime/live-evidence.json").write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))
