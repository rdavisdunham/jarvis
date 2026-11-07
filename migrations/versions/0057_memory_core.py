"""Core memories: facts the owner confirmed Dani should always keep in mind (Set up with Dani, About you)."""
import sqlalchemy as sa
from alembic import op

revision = "0057_memory_core"
down_revision = "0056_setup_sessions"
branch_labels = depends_on = None


def upgrade():
    op.add_column("memory_assertions", sa.Column("core", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.create_index("ix_memory_assertions_core", "memory_assertions", ["core"])


def downgrade():
    op.drop_index("ix_memory_assertions_core", "memory_assertions")
    op.drop_column("memory_assertions", "core")
