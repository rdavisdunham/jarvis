"""The owner's "About you" summary for voice sessions (about_you.py)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0055_about_you"
down_revision = "0054_person_bucket_links"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "about_you_summaries",
        sa.Column("owner_id", sa.String(100), primary_key=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("memory_ids", postgresql.JSONB(), nullable=False),
        sa.Column("model", sa.String(80)),
        sa.Column("prompt_version", sa.String(40)),
        sa.Column("cost", sa.Float()),
        sa.Column("generated_at", sa.DateTime(timezone=True)),
        sa.Column("checked_at", sa.DateTime(timezone=True)),
        sa.Column("next_check_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", postgresql.JSONB()),
    )
    op.create_index("ix_about_you_summaries_next_check_at", "about_you_summaries", ["next_check_at"])


def downgrade():
    op.drop_table("about_you_summaries")
