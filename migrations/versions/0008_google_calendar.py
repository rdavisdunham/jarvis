"""Owner-bound Google sign-in and outbound read-only calendar sync."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008_google_calendar"
down_revision = "0007_notes_context"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "auth_sessions", sa.Column("auth_method", sa.String(20), nullable=False, server_default="pairing")
    )
    op.create_table(
        "google_identities",
        sa.Column("owner_id", sa.String(100), primary_key=True),
        sa.Column("subject", sa.String(255), nullable=False, unique=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("credentials", sa.Text()),
        sa.Column("calendar_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("generation", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(30), nullable=False, server_default="not_connected"),
        sa.Column("last_sync_at", sa.DateTime(timezone=True)),
        sa.Column("next_sync_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.String(100), nullable=False, server_default=""),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "google_oauth_attempts",
        sa.Column("state_hash", sa.String(64), primary_key=True),
        sa.Column("browser_hash", sa.String(64), nullable=False),
        sa.Column("purpose", sa.String(20), nullable=False),
        sa.Column("account_subject", sa.String(255)),
        sa.Column("account_generation", sa.Integer()),
        sa.Column("session_hash", sa.String(64)),
        sa.Column("nonce", sa.String(100), nullable=False),
        sa.Column("verifier", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "google_calendars",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "owner_id",
            sa.String(100),
            sa.ForeignKey("google_identities.owner_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("provider_id", sa.String(1024), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("timezone", sa.String(100), nullable=False, server_default="UTC"),
        sa.Column("selected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("available", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("primary", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("sync_token", sa.Text()),
        sa.Column("last_sync_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("owner_id", "provider_id", name="uq_google_calendar"),
    )
    op.create_table(
        "google_calendar_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "calendar_id", sa.String(36), sa.ForeignKey("google_calendars.id"), nullable=False, index=True
        ),
        sa.Column("provider_id", sa.String(1024), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.UniqueConstraint("calendar_id", "provider_id", name="uq_google_event"),
    )


def downgrade():
    for table in ["google_calendar_events", "google_calendars", "google_oauth_attempts", "google_identities"]:
        op.drop_table(table)
    op.drop_column("auth_sessions", "auth_method")
