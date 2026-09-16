"""Short-lived database bridge between background workers and the active browser."""

import asyncio
from datetime import timedelta

from sqlalchemy import select

from .access import actor, execution, identity
from .db import session_scope
from .domain import DomainError
from .models import AuthSession, DeviceAction, DeviceBridge, now
from .work_crypto import seal, unseal


def current_device(db, owner, device, account):
    sessions = list(
        db.scalars(
            select(AuthSession).where(
                AuthSession.device_id == device,
                AuthSession.owner_id == account,
                AuthSession.expires_at > now(),
            )
        )
    )
    for session in sessions:
        try:
            namespace, _ = identity(db, session)
            if namespace == owner:
                return True
        except DomainError:
            pass
    return False


def context(owner, device):
    grant = execution.get()
    if not grant:
        return {}
    with session_scope() as db:
        row = db.get(DeviceBridge, (owner, device))
        if (
            not row
            or row.account_id != grant["account_id"]
            or row.expires_at <= now()
            or not current_device(db, owner, device, row.account_id)
        ):
            return {}
        return unseal(row.context_ciphertext)


def sync(owner, device, body):
    with session_scope() as db:
        account = actor(db, owner)
        if not current_device(db, owner, device, account):
            return []
        row = db.get(DeviceBridge, (owner, device))
        if not row:
            row = DeviceBridge(owner_id=owner, device_id=device, account_id=account)
            db.add(row)
        row.context_ciphertext = seal(body.context.model_dump())
        row.expires_at = now() + timedelta(seconds=60)
        for result in body.results:
            action = db.get(DeviceAction, (owner, result["id"]))
            if (
                action
                and action.device_id == device
                and action.account_id == account
                and not action.result_ciphertext
                and action.expires_at > now()
            ):
                action.result_ciphertext = seal({**result, "screen": body.context.model_dump()})
        # Only one action at a time on a device; the next can appear after acknowledgement.
        action = db.scalar(
            select(DeviceAction)
            .where(
                DeviceAction.owner_id == owner,
                DeviceAction.device_id == device,
                DeviceAction.account_id == account,
                DeviceAction.result_ciphertext.is_(None),
                DeviceAction.expires_at > now(),
            )
            .order_by(DeviceAction.created_at, DeviceAction.id)
            .limit(1)
        )
        if not action:
            return []
        action.sent_at = action.sent_at or now()
        return [unseal(action.action_ciphertext)]


async def dispatch(owner, device, action):
    account = execution.get()["account_id"]
    with session_scope() as db:
        if not current_device(db, owner, device, account):
            return {"status": "failed", "message": "Open the app in this workspace to change its screen."}
        existing = db.get(DeviceAction, (owner, action["id"]))
        if existing:
            if existing.result_ciphertext:
                return {"ui_action": action, **unseal(existing.result_ciphertext)}
            if existing.sent_at:
                return {
                    "status": "failed",
                    "message": "An earlier screen action has an uncertain outcome. Check the current screen.",
                }
        else:
            db.add(
                DeviceAction(
                    owner_id=owner,
                    id=action["id"],
                    device_id=device,
                    account_id=account,
                    action_ciphertext=seal(action),
                    expires_at=now() + timedelta(seconds=12),
                )
            )
    for _ in range(40):
        await asyncio.sleep(0.25)
        with session_scope() as db:
            row = db.get(DeviceAction, (owner, action["id"]))
            if not row or row.expires_at <= now() or not current_device(db, owner, device, account):
                break
            if row.result_ciphertext:
                return {"ui_action": action, **unseal(row.result_ciphertext)}
    with session_scope() as db:
        row = db.get(DeviceAction, (owner, action["id"]))
        if row:
            row.expires_at = now()
    return {
        "status": "failed",
        "message": "The current screen did not acknowledge this action. Try again with the app open.",
    }


def cleanup(db):
    for model in (DeviceAction, DeviceBridge):
        retention = timedelta(hours=24) if model is DeviceAction else timedelta(minutes=2)
        for row in db.scalars(select(model).where(model.expires_at < now() - retention).limit(200)):
            db.delete(row)
