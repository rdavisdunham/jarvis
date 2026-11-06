"""Living overviews for clients and projects (record_overviews.py)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0052_record_overviews"
down_revision = "0051_linear_comments"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "record_overviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("record_id", sa.String(36), sa.ForeignKey("structure_records.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("sections", postgresql.JSONB(), nullable=False),
        sa.Column("seed", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("origin", sa.String(20), nullable=False),
        sa.Column("stats", postgresql.JSONB(), nullable=False),
        sa.Column("model", sa.String(80)),
        sa.Column("prompt_version", sa.String(40)),
        sa.Column("cost", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("record_id", "version"),
    )
    op.create_index("ix_record_overviews_owner_record", "record_overviews", ["owner_id", "record_id"])
    op.create_table(
        "record_overview_states",
        sa.Column("record_id", sa.String(36), sa.ForeignKey("structure_records.id", ondelete="CASCADE"),
                  primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("seen", postgresql.JSONB(), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True)),
        sa.Column("next_check_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", postgresql.JSONB()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_record_overview_states_owner_id", "record_overview_states", ["owner_id"])
    op.create_index("ix_record_overview_states_next_check_at", "record_overview_states", ["next_check_at"])


def downgrade():
    op.drop_table("record_overview_states")
    op.drop_table("record_overviews")
