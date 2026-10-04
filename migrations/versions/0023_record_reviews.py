"""Review cadence: per-record last/next review, a per-record pause and the daily inbox marker."""

from alembic import op
import sqlalchemy as sa

revision = "0023_record_reviews"
down_revision = "0022_record_templates"
branch_labels = depends_on = None


def upgrade():
    op.add_column("structure_records", sa.Column("last_reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("structure_records", sa.Column("next_review_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("structure_records", sa.Column("review_queued_at", sa.DateTime(timezone=True), nullable=True))
    # Previous-version writers omit the flag; the server default keeps their inserts readable.
    op.add_column(
        "structure_records",
        sa.Column("review_paused", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index(
        "ix_structure_records_owner_id_next_review_at", "structure_records", ["owner_id", "next_review_at"]
    )


def downgrade():
    op.drop_index("ix_structure_records_owner_id_next_review_at", table_name="structure_records")
    for column in ("review_paused", "review_queued_at", "next_review_at", "last_reviewed_at"):
        op.drop_column("structure_records", column)
