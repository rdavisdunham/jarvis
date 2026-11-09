"""Remotea sessions stay findable after their turns go, and chats from outside remotea leave the ledger.

`agent_sessions.recall`: a longer record of what was discussed, written from the session's turns by the summary job
and carried forward on each re-summary (turns of a session no run owns are now kept for the owner's run window,
90 days by default, instead of 30).

Data: every session that never ran in remotea (no surface, state, run or host, and no `live` or `remotea open`
resume from the collector) is archived, at the owner's request (2026-11-09): the collector now reports remotea
sessions only. Archived sessions leave every listing and search; the archive endpoint
(`POST /sessions/{id}/archive` with `archived: false`) restores one. The downgrade leaves them archived: the
archived flag cannot tell these apart from sessions the owner hid by hand."""
import sqlalchemy as sa
from alembic import op

revision = "0061_session_recall"
down_revision = "0060_turn_search"
branch_labels = depends_on = None

OUTSIDE_REMOTEA = """
    archived = false AND surf = '' AND state = '' AND run_id IS NULL AND host_id IS NULL
    AND NOT (coalesce(resume, '{}'::jsonb) ? 'live')
    AND coalesce(resume ->> 'cmd', '') NOT LIKE 'remotea open %'
"""


def upgrade():
    op.add_column("agent_sessions", sa.Column("recall", sa.Text(), nullable=False, server_default=""))
    op.execute(f"UPDATE agent_sessions SET archived = true, updated_at = now() WHERE {OUTSIDE_REMOTEA}")


def downgrade():
    op.drop_column("agent_sessions", "recall")
