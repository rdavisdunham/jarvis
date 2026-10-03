"""Provider provenance is integration metadata, never a user classification."""
from .calendar_details import web_link

def compact_source(source):
    return {k:source.get(k) for k in ("provider","label","context","item_id","sync_state","url")}

def native(identity=None):
    return {"provider":"eridani", "label":"Eridani", "item_id":identity,
            "sync_state":"local", "context":"This workspace", "url":None,
            "editable_fields":[], "read_only_fields":[], "local_only_fields":["local_notes", "main_home", "custom_fields"]}

def task_source(db, task):
    from .linear_sync import link_for_task
    from .models import LinearConnection
    link = link_for_task(db, task.owner_id, task.id)
    if not link:
        return native(task.id)
    conn = db.get(LinearConnection, task.owner_id)
    snap = link.snapshot or {}
    state = link.sync_state if conn and conn.enabled else "disconnected"
    reason = ("This issue is unlinked; edits stay in Eridani." if state == "unlinked" else
              "Reconnect this Linear workspace to edit synced fields." if not conn or not conn.enabled or conn.workspace_id != link.workspace_id else
              "Select this issue's team in Settings to edit synced fields." if (snap.get("team") or {}).get("id") not in conn.team_ids else "")
    return {**native(task.id), "provider":"linear", "label":"Linear",
            "account_id":link.workspace_id, "container_id":(snap.get("team") or {}).get("id"),
            "item_id":link.remote_id, "identifier":snap.get("identifier"),
            "context":" · ".join(filter(None, [conn.workspace_name if conn else None, (snap.get("team") or {}).get("name")])),
            "url":web_link(snap.get("url")), "sync_state":state,
            "remote_revision":snap.get("updatedAt"),
            "editable_fields":["title","description","status","priority","due_date","assignee","remote_project"] if not reason else [],
            "read_only_fields":["identifier","team","labels","parent","updatedAt"],
            "local_only_fields":["local_notes","main_home","custom_fields","due_time","planned_date","estimate_minutes","archive"],
            "read_only_reason":reason,
            "details":{k:snap[k] for k in ("identifier","team","project","assignee","state","priority","dueDate","parent","labels","updatedAt","archivedAt") if k in snap}}

def google_source(db, calendar, payload, state="synced"):
    from .models import GoogleIdentity, Job
    from sqlalchemy import select
    from .google_writes import blocked_event
    account=db.get(GoogleIdentity,calendar.owner_id)
    enabled=bool(account and account.calendar_enabled)
    if state == "synced" and payload.get("id"):
        job=db.scalar(select(Job).where(Job.owner_id==calendar.owner_id,Job.kind=="google_write",
            Job.payload["provider_calendar"].astext==calendar.provider_id,
            Job.payload["provider_event"].astext==payload["id"]).order_by(Job.created_at.desc()).limit(1))
        if job and job.status not in {"succeeded","cancelled"}:
            state=job.status
    reason=blocked_event(payload)
    writable=bool(not reason and enabled and account.calendar_write_enabled and calendar.available and calendar.access_role in {"owner","writer"})
    return {**native(), "provider":"google", "label":"Google Calendar", "account_id":account.subject if account else None,
            "container_id":calendar.provider_id,"item_id":payload.get("id"),
            "context":" · ".join(filter(None,[account.email if account else None,calendar.title])),
            "url":web_link(payload.get("htmlLink")), "sync_state":state if enabled else "disconnected",
            "remote_revision":payload.get("etag"), "series_id":payload.get("recurringEventId"),
            "occurrence_start":payload.get("originalStartTime"),
            "editable_fields":["title","start","end","all_day","timezone","location","description","busy"] if writable else [],
            "read_only_fields":["organizer","attendees","attachments","meeting_url"],
            "read_only_reason":"" if writable else reason or "Calendar is read-only or Google write access is disconnected."}
