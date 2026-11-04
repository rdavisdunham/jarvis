"""Context REST routes. `router` is mounted on the external (bot-key) router in `external_routes.py` under
/api/v1/external; `session_router` holds browser (session cookie) routes with their own /api/v1/context prefix
and is registered in `api.py`.

Each context package adds one `include_router(...)` line here.
"""

from fastapi import APIRouter

from . import admin, sources

router = APIRouter()
router.include_router(sources.router)

session_router = APIRouter()
session_router.include_router(admin.router)
