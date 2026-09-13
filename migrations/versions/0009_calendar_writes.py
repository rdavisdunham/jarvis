"""Optional Google event writes and calendar access roles."""

import sqlalchemy as sa
from alembic import op

revision = "0009_calendar_writes"
down_revision = "0008_google_calendar"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "google_identities",
        sa.Column("calendar_write_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "google_calendars", sa.Column("access_role", sa.String(40), nullable=False, server_default="reader")
    )


def downgrade():
    op.drop_column("google_calendars", "access_role")
    op.drop_column("google_identities", "calendar_write_enabled")
