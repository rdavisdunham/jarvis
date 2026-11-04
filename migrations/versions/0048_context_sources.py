"""Context platform sources: connections, sources, immutable revisions, segments, chunks, runs, cursors.

Embeddings are pgvector vector(512) projections (production runs the pgvector image; see docs/CLOUD_MIGRATION.md).
Exact cosine search for now; an HNSW index waits for measurement (0053_context_hnsw).
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0048_context_sources"
down_revision = "0047_actor_name_uniqueness"
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


def tsv(column):
    return sa.Column("tsv", postgresql.TSVECTOR(), sa.Computed(f"to_tsvector('english', {column})", persisted=True),
                     nullable=False)


def owner_index(table):
    op.create_index(f"ix_{table}_owner_id", table, ["owner_id"])


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "context_connections", *ident(),
        sa.Column("connector", sa.String(20), nullable=False),
        sa.Column("external_account", sa.String(320), nullable=False),
        sa.Column("credential_ciphertext", sa.Text(), nullable=True),
        jsonb("checkpoint"),
        sa.Column("health", sa.String(20), nullable=False, server_default="unknown"),
        sa.Column("last_error", sa.Text(), nullable=True),
        at("last_success_at"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        *stamps(),
        sa.UniqueConstraint("owner_id", "connector", "external_account", name="uq_context_connections_account"),
    )
    owner_index("context_connections")
    op.create_table(
        "context_sources", *ident(),
        sa.Column("connection_id", sa.String(36), sa.ForeignKey("context_connections.id"), nullable=True),
        sa.Column("connector", sa.String(20), nullable=False),
        sa.Column("external_id", sa.String(300), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False, server_default="meeting"),
        sa.Column("title", sa.String(500), nullable=False, server_default=""),
        sa.Column("upstream_owner_email", sa.String(320), nullable=True),
        sa.Column("upstream_owner_name", sa.String(200), nullable=True),
        sa.Column("current_revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="received"),
        sa.Column("availability", sa.String(20), nullable=False, server_default="available"),
        at("deleted_at"),
        *stamps(),
        sa.UniqueConstraint("owner_id", "connector", "external_id", name="uq_context_sources_external"),
    )
    owner_index("context_sources")
    op.create_index("ix_context_sources_connection_id", "context_sources", ["connection_id"])
    op.create_table(
        "context_source_revisions", *ident(),
        sa.Column("source_id", sa.String(36), sa.ForeignKey("context_sources.id"), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("external_revision", sa.String(200), nullable=True),
        sa.Column("change_kind", sa.String(20), nullable=False, server_default="content"),
        jsonb("payload"),
        at("occurred_at"),
        at("ended_at"),
        sa.Column("timezone", sa.String(64), nullable=True),
        sa.Column("calendar_event_id", sa.String(300), nullable=True),
        at("upstream_updated_at"),
        jsonb("participants", "[]"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        *stamps(updated=False),
        sa.UniqueConstraint("source_id", "revision", name="uq_context_source_revisions_revision"),
    )
    owner_index("context_source_revisions")
    op.create_table(
        "context_segments", *ident(),
        sa.Column("revision_id", sa.String(36), sa.ForeignKey("context_source_revisions.id"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False, server_default="transcript"),
        sa.Column("speaker_label", sa.String(200), nullable=True),
        sa.Column("speaker_actor_id", sa.String(36), nullable=True),
        sa.Column("speaker_link_status", sa.String(20), nullable=False, server_default="unresolved"),
        jsonb("speaker_provenance"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=True),
        sa.Column("end_ms", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        tsv("text"),
        *stamps(updated=False),
        sa.UniqueConstraint("revision_id", "seq", name="uq_context_segments_seq"),
    )
    owner_index("context_segments")
    op.create_index("ix_context_segments_tsv", "context_segments", ["tsv"], postgresql_using="gin")
    op.create_table(
        "context_chunks", *ident(),
        sa.Column("revision_id", sa.String(36), sa.ForeignKey("context_source_revisions.id"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("segment_start", sa.Integer(), nullable=False),
        sa.Column("segment_end", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(512), nullable=True),
        sa.Column("embedding_model", sa.String(100), nullable=True),
        at("embedded_at"),
        tsv("text"),
        *stamps(),
        sa.UniqueConstraint("revision_id", "seq", name="uq_context_chunks_seq"),
    )
    owner_index("context_chunks")
    op.create_index("ix_context_chunks_tsv", "context_chunks", ["tsv"], postgresql_using="gin")
    op.create_table(
        "context_runs", *ident(),
        sa.Column("stage", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("job_id", sa.String(36), nullable=True),
        sa.Column("source_id", sa.String(36), sa.ForeignKey("context_sources.id"), nullable=True),
        sa.Column("revision_id", sa.String(36), sa.ForeignKey("context_source_revisions.id"), nullable=True),
        sa.Column("entity_type", sa.String(40), nullable=True),
        sa.Column("entity_id", sa.String(100), nullable=True),
        sa.Column("engine", sa.String(60), nullable=False, server_default="native"),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("prompt_version", sa.String(60), nullable=True),
        sa.Column("config_version", sa.String(60), nullable=True),
        at("decision_watermark"),
        sa.Column("epoch", sa.Integer(), nullable=False, server_default="0"),
        jsonb("coverage"),
        jsonb("stage_results"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        at("started_at"),
        at("finished_at"),
        *stamps(),
    )
    owner_index("context_runs")
    op.create_index("ix_context_runs_source_id", "context_runs", ["source_id"])
    op.create_index("ix_context_runs_owner_stage_status", "context_runs", ["owner_id", "stage", "status"])
    op.create_table(
        "context_cursors", *ident(),
        sa.Column("connection_id", sa.String(36), sa.ForeignKey("context_connections.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        jsonb("value"),
        *stamps(),
        sa.UniqueConstraint("connection_id", "name", name="uq_context_cursors_name"),
    )
    owner_index("context_cursors")


def downgrade():
    # Dropping a table drops its indexes; the vector extension stays (other database objects may use it).
    for table in ("context_cursors", "context_runs", "context_chunks", "context_segments", "context_source_revisions",
                  "context_sources", "context_connections"):
        op.drop_table(table)
