"""Source-backed automatic memory with cloud embeddings.

Revision ID: 0003_memory_learning
Revises: 0002_reminder_completion
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003_memory_learning"
down_revision = "0002_reminder_completion"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("sources", sa.Column("memory_version", sa.Integer(), nullable=False, server_default="0"))
    for name, type_, default in [
        ("tags", JSONB(), "'[]'::jsonb"),
        ("evidence", sa.Text(), "''"),
        ("fact_key", sa.String(200), "''"),
        ("fingerprint", sa.String(64), "''"),
    ]:
        op.add_column(
            "memory_assertions", sa.Column(name, type_, nullable=False, server_default=sa.text(default))
        )
    op.add_column("memory_assertions", sa.Column("embedding", JSONB(), nullable=True))
    op.add_column("memory_assertions", sa.Column("embedding_model", sa.String(100), nullable=True))
    op.create_index("ix_memory_assertions_fingerprint", "memory_assertions", ["fingerprint"])


def downgrade():
    op.drop_index("ix_memory_assertions_fingerprint", "memory_assertions")
    for name in ["embedding_model", "embedding", "fingerprint", "fact_key", "evidence", "tags"]:
        op.drop_column("memory_assertions", name)
    op.drop_column("sources", "memory_version")
