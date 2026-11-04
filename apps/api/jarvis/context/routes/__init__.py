"""Context REST routes.

`router` is mounted on the external (bot-key) router in `external_routes.py` under /api/v1/external. Each context
package adds one `router.include_router(...)` line here.

`owner_router` holds routes for the signed-in owner's browser session under /api/v1/context. It must never be
mounted under /api/v1/external, whose responses the security middleware re-authenticates as bot keys; `api.py`
mounts it with `app.include_router(owner_router)`.
"""

from fastapi import APIRouter

from . import admin, context, owner, review, settings, sources

router = APIRouter()
router.include_router(sources.router)
router.include_router(context.router)
router.include_router(review.external)

owner_router = APIRouter(prefix="/api/v1/context")
owner_router.include_router(settings.router)
owner_router.include_router(admin.router)
owner_router.include_router(owner.router)
owner_router.include_router(review.router)
