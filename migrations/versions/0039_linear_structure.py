"""Linear structure onboarding: the owner's team/project → Home mapping on the connection (linear_structure.py)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0039_linear_structure"
down_revision = "0038_memory_review_details"
branch_labels = depends_on = None


def upgrade():
    op.add_column(
        "linear_connections",
        sa.Column("structure", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )


def downgrade():
    op.drop_column("linear_connections", "structure")
