"""Context platform interpretation: decisions, assertions and their versions, evidence, entity links, actor
identities and merges. Eridani entities are (entity_type, entity_id) strings, never foreign keys."""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0049_context_assertions"
down_revision = "0048_context_sources"
branch_labels = depends_on = None


def ident():
    return [sa.Column("id", sa.String(36), primary_key=True), sa.Column("owner_id", sa.String(100), nullable=False)]


def stamps(updated=True):
    names = ("created_at", "updated_at") if updated else ("created_at",)
    return [sa.Column(name, sa.DateTime(timezone=True), nullable=False) for name in names]


def jsonb(name, empty="{}"):
    return sa.Column(name, postgresql.JSONB(), nullable=False, server_default=empty)


def at(name):
    return sa.Column(name, sa.DateTime(timezone=True), nullable=True)


def ref(name, table, nullable=True):
    return sa.Column(name, sa.String(36), sa.ForeignKey(f"{table}.id"), nullable=nullable)


def owner_index(table):
    op.create_index(f"ix_{table}_owner_id", table, ["owner_id"])


def upgrade():
    op.create_table(
        "context_decisions", *ident(),
        sa.Column("target_type", sa.String(40), nullable=False),
        sa.Column("target_id", sa.String(100), nullable=False),
        sa.Column("expected_revision", sa.Integer(), nullable=True),
        sa.Column("operation", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("negative", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("origin", sa.String(20), nullable=False),
        sa.Column("principal", sa.String(100), nullable=False),
        sa.Column("command_id", sa.String(100), nullable=True),
        jsonb("before"),
        jsonb("after"),
        at("effective_at"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        ref("undoes_id", "context_decisions"),
        *stamps(updated=False),
    )
    owner_index("context_decisions")
    op.create_index("ix_context_decisions_command_id", "context_decisions", ["command_id"])
    op.create_index("ix_context_decisions_target", "context_decisions", ["owner_id", "target_type", "target_id"])
    op.create_table(
        "context_assertions", *ident(),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("subject_type", sa.String(40), nullable=False),
        sa.Column("subject_id", sa.String(100), nullable=False),
        sa.Column("object_type", sa.String(40), nullable=True),
        sa.Column("object_id", sa.String(100), nullable=True),
        sa.Column("predicate", sa.String(100), nullable=False),
        sa.Column("key", sa.String(300), nullable=True),
        sa.Column("modality", sa.String(20), nullable=False, server_default="asserted"),
        sa.Column("explicit", sa.Boolean(), nullable=False, server_default="true"),
        at("valid_from"),
        at("valid_to"),
        sa.Column("valid_precision", sa.String(20), nullable=False, server_default="unknown"),
        sa.Column("lifecycle", sa.String(20), nullable=False, server_default="current"),
        sa.Column("current_version", sa.Integer(), nullable=False, server_default="1"),
        ref("supersedes_id", "context_assertions"),
        ref("run_id", "context_runs"),
        sa.Column("confidence", sa.Float(), nullable=True),
        jsonb("responsible", "[]"),
        sa.Column("recipient", sa.String(100), nullable=True),
        at("due_at"),
        sa.Column("due_precision", sa.String(20), nullable=True),
        jsonb("completion_evidence", "[]"),
        sa.Column("embedding", Vector(512), nullable=True),
        sa.Column("embedding_model", sa.String(100), nullable=True),
        at("embedded_at"),
        sa.Column("tsv", postgresql.TSVECTOR(),
                  sa.Computed("to_tsvector('english', statement)", persisted=True), nullable=False),
        *stamps(),
    )
    owner_index("context_assertions")
    op.create_index("ix_context_assertions_subject", "context_assertions", ["owner_id", "subject_type", "subject_id"])
    op.create_index("ix_context_assertions_lifecycle", "context_assertions", ["owner_id", "lifecycle"])
    op.create_index("ix_context_assertions_tsv", "context_assertions", ["tsv"], postgresql_using="gin")
    op.create_table(
        "context_assertion_versions", *ident(),
        ref("assertion_id", "context_assertions", nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("lifecycle", sa.String(20), nullable=False),
        at("valid_from"),
        at("valid_to"),
        sa.Column("valid_precision", sa.String(20), nullable=False, server_default="unknown"),
        sa.Column("recorded_from", sa.DateTime(timezone=True), nullable=False),
        at("recorded_to"),
        ref("decision_id", "context_decisions"),
        ref("run_id", "context_runs"),
        *stamps(updated=False),
        sa.UniqueConstraint("assertion_id", "version", name="uq_context_assertion_versions_version"),
    )
    owner_index("context_assertion_versions")
    op.create_table(
        "context_evidence", *ident(),
        ref("assertion_id", "context_assertions", nullable=False),
        sa.Column("assertion_version", sa.Integer(), nullable=False),
        ref("source_id", "context_sources", nullable=False),
        ref("revision_id", "context_source_revisions", nullable=False),
        ref("segment_id", "context_segments"),
        sa.Column("quote", sa.Text(), nullable=False, server_default=""),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(20), nullable=False, server_default="support"),
        sa.Column("origin", sa.String(20), nullable=False, server_default="independent"),
        jsonb("derivation"),
        *stamps(updated=False),
    )
    owner_index("context_evidence")
    for column in ("assertion_id", "source_id", "segment_id"):
        op.create_index(f"ix_context_evidence_{column}", "context_evidence", [column])
    op.create_table(
        "context_entity_links", *ident(),
        ref("assertion_id", "context_assertions"),
        sa.Column("subject_type", sa.String(40), nullable=False),
        sa.Column("subject_id", sa.String(100), nullable=False),
        sa.Column("target_type", sa.String(40), nullable=True),
        sa.Column("target_id", sa.String(100), nullable=True),
        sa.Column("mention", sa.String(500), nullable=False, server_default=""),
        sa.Column("role", sa.String(40), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="unresolved"),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("margin", sa.Float(), nullable=True),
        jsonb("evidence", "[]"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        ref("decision_id", "context_decisions"),
        ref("run_id", "context_runs"),
        *stamps(),
    )
    owner_index("context_entity_links")
    op.create_index("ix_context_entity_links_assertion_id", "context_entity_links", ["assertion_id"])
    op.create_index("ix_context_entity_links_subject", "context_entity_links", ["owner_id", "subject_type", "subject_id"])
    op.create_index("ix_context_entity_links_target", "context_entity_links", ["owner_id", "target_type", "target_id"])
    op.create_table(
        "context_actor_identities", *ident(),
        sa.Column("actor_id", sa.String(36), nullable=False),
        sa.Column("namespace", sa.String(40), nullable=False),
        sa.Column("value", sa.String(500), nullable=False),
        sa.Column("normalized", sa.String(500), nullable=False),
        sa.Column("source_scope", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="inferred"),
        jsonb("provenance"),
        *stamps(),
    )
    owner_index("context_actor_identities")
    op.create_index("ix_context_actor_identities_actor_id", "context_actor_identities", ["actor_id"])
    # One row per actor and identifier; a strong identifier names at most one actor. NULL source_scope is global.
    key = ["owner_id", "namespace", "normalized", "source_scope"]
    op.create_index("uq_context_actor_identities_actor", "context_actor_identities", [*key, "actor_id"], unique=True,
                    postgresql_nulls_not_distinct=True)
    op.create_index("uq_context_actor_identities_strong", "context_actor_identities", key, unique=True,
                    postgresql_nulls_not_distinct=True,
                    postgresql_where=sa.text("namespace NOT IN ('alias', 'calendar_name')"))
    op.create_table(
        "context_actor_merges", *ident(),
        sa.Column("winner_actor_id", sa.String(36), nullable=False),
        sa.Column("loser_actor_id", sa.String(36), nullable=False),
        jsonb("snapshot"),
        ref("decision_id", "context_decisions"),
        at("undone_at"),
        *stamps(),
    )
    owner_index("context_actor_merges")


def downgrade():
    for table in ("context_actor_merges", "context_actor_identities", "context_entity_links", "context_evidence",
                  "context_assertion_versions", "context_assertions", "context_decisions"):
        op.drop_table(table)
