"""Context platform review: task proposals, publications, entity briefs and per-owner processing state."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0050_context_review"
down_revision = "0049_context_assertions"
branch_labels = depends_on = None


def ident():
    return [sa.Column("id", sa.String(36), primary_key=True), sa.Column("owner_id", sa.String(100), nullable=False)]


def stamps():
    return [sa.Column(name, sa.DateTime(timezone=True), nullable=False) for name in ("created_at", "updated_at")]


def jsonb(name, empty="{}"):
    return sa.Column(name, postgresql.JSONB(), nullable=False, server_default=empty)


def ref(name, table, nullable=True):
    return sa.Column(name, sa.String(36), sa.ForeignKey(f"{table}.id"), nullable=nullable)


def owner_index(table):
    op.create_index(f"ix_{table}_owner_id", table, ["owner_id"])


def upgrade():
    op.create_table(
        "context_proposals", *ident(),
        ref("source_id", "context_sources"),
        ref("revision_id", "context_source_revisions"),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        jsonb("payload"),
        sa.Column("destination", sa.String(20), nullable=False, server_default="eridani"),
        sa.Column("responsible", sa.String(36), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        jsonb("assertion_ids", "[]"),
        jsonb("evidence", "[]"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("disposition", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("preselected", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("task_id", sa.String(36), nullable=True),
        *stamps(),
        # Lint proposals have no source; NULLS NOT DISTINCT keeps their fingerprints unique too.
        sa.UniqueConstraint("owner_id", "source_id", "fingerprint", name="uq_context_proposals_fingerprint",
                            postgresql_nulls_not_distinct=True),
    )
    owner_index("context_proposals")
    op.create_index("ix_context_proposals_disposition", "context_proposals", ["owner_id", "disposition"])
    op.create_table(
        "context_publications", *ident(),
        ref("proposal_id", "context_proposals", nullable=False),
        sa.Column("proposal_revision", sa.Integer(), nullable=False),
        jsonb("payload"),
        sa.Column("destination", sa.String(20), nullable=False),
        sa.Column("principal", sa.String(100), nullable=False),
        sa.Column("command_id", sa.String(100), nullable=True),
        sa.Column("operation_id", sa.String(100), nullable=False, unique=True),
        sa.Column("task_id", sa.String(36), nullable=True),
        sa.Column("remote_issue_id", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        jsonb("outcome"),
        *stamps(),
    )
    owner_index("context_publications")
    op.create_index("ix_context_publications_proposal_id", "context_publications", ["proposal_id"])
    op.create_table(
        "context_briefs", *ident(),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(100), nullable=False),
        jsonb("sections", "[]"),
        jsonb("dependencies"),
        jsonb("watermarks"),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("stale", sa.Boolean(), nullable=False, server_default="false"),
        ref("run_id", "context_runs"),
        *stamps(),
        sa.UniqueConstraint("owner_id", "entity_type", "entity_id", "version", name="uq_context_briefs_version"),
    )
    owner_index("context_briefs")
    op.create_table(
        "context_owner_state",
        sa.Column("owner_id", sa.String(100), primary_key=True),
        sa.Column("epoch", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("decision_seq", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("paused", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        jsonb("settings"),
        *stamps(),
    )


def downgrade():
    for table in ("context_owner_state", "context_briefs", "context_publications", "context_proposals"):
        op.drop_table(table)
