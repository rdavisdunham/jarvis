"""Authored notes, revisioned search chunks and task references."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007_notes_context"
down_revision = "0006_workspace_accounting"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "notes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False, index=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("tags", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id"), index=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id")),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("index_state", sa.String(30), nullable=False, server_default="queued"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "note_embeddings",
        sa.Column("note_id", sa.String(36), sa.ForeignKey("notes.id"), primary_key=True),
        sa.Column("position", sa.Integer(), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("embedding", JSONB(), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
    )
    op.create_table(
        "note_task_links",
        sa.Column("note_id", sa.String(36), sa.ForeignKey("notes.id"), primary_key=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id"), primary_key=True),
        sa.Column("linked", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("evidence", sa.Text(), nullable=False, server_default=""),
        sa.Column("note_revision", sa.Integer()),
        sa.Column("fingerprint", sa.String(64)),
        sa.UniqueConstraint("note_id", "fingerprint", name="uq_note_extraction"),
    )
    op.create_table(
        "task_references",
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), primary_key=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id"), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False, index=True),
        sa.Column("touched_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade():
    for name in ["task_references", "note_task_links", "note_embeddings", "notes"]:
        op.drop_table(name)
