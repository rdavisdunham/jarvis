"""People linked to clients and projects (people_links.py, organization knowledge phase 4)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0054_person_bucket_links"
down_revision = "0053_linear_placement_evidence"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "person_bucket_links",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("actor_id", sa.String(36), sa.ForeignKey("actors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("record_id", sa.String(36), sa.ForeignKey("structure_records.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("tasks", sa.Integer(), nullable=False),
        sa.Column("meeting_facts", sa.Integer(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("actor_id", "record_id"),
    )
    op.create_index("ix_person_bucket_links_owner_id", "person_bucket_links", ["owner_id"])
    op.create_index("ix_person_bucket_links_actor_id", "person_bucket_links", ["actor_id"])
    op.create_index("ix_person_bucket_links_owner_record", "person_bucket_links", ["owner_id", "record_id"])


def downgrade():
    op.drop_table("person_bucket_links")
