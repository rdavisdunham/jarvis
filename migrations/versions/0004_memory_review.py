"""Weekly memory review and provenance-preserving merge links."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004_memory_review"
down_revision = "0003_memory_learning"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("memory_assertions", sa.Column("merged_into_id", sa.String(36), nullable=True))
    op.create_foreign_key(
        "fk_memory_merged_into",
        "memory_assertions",
        "memory_assertions",
        ["merged_into_id"],
        ["id"],
    )
    op.create_table(
        "memory_reviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False, index=True),
        sa.Column("pair_key", sa.String(64), nullable=False),
        sa.Column("memory_ids", JSONB, nullable=False),
        sa.Column("memory_revisions", JSONB, nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("result_memory_id", sa.String(36), sa.ForeignKey("memory_assertions.id")),
        sa.Column("last_offered_at", sa.DateTime(timezone=True)),
        sa.Column("deferred_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("owner_id", "pair_key", name="uq_memory_review_pair"),
    )


def downgrade():
    op.drop_table("memory_reviews")
    op.drop_constraint("fk_memory_merged_into", "memory_assertions", type_="foreignkey")
    op.drop_column("memory_assertions", "merged_into_id")
