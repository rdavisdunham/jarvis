"""Context REST routes, mounted on the external (bot-key) router in `external_routes.py` under /api/v1/external.

Each context package adds one `router.include_router(...)` line here.
"""

from fastapi import APIRouter

from . import context, sources

router = APIRouter()
router.include_router(sources.router)
router.include_router(context.router)
