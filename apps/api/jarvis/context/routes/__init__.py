"""Context REST routes. `router` is mounted on the external (bot-key) router in `external_routes.py` under
/api/v1/external; `owner_router` holds browser (session cookie) routes under /api/v1/context and is mounted in
`api.py`.

Each context package adds one `include_router(...)` line here.
"""

from fastapi import APIRouter

from . import admin, sources

router = APIRouter()
router.include_router(sources.router)

owner_router = APIRouter(prefix="/api/v1/context")
owner_router.include_router(admin.router)
