"""Bug reports from the web app's Report a bug button: description, context, screenshot and the linked task."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0042_bug_reports"
down_revision = "0041_hyperplan"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "bug_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("expected", sa.Text(), nullable=False, server_default=""),
        sa.Column("context", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("screenshot", sa.LargeBinary(), nullable=True),
        sa.Column("screenshot_mime", sa.String(40), nullable=True),
        sa.Column("screenshot_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("screenshot_error", sa.String(300), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("task_id", sa.String(36), nullable=True),
    )
    op.create_index("ix_bug_reports_owner_id", "bug_reports", ["owner_id"])
    op.create_index("ix_bug_reports_task_id", "bug_reports", ["task_id"])


def downgrade():
    op.drop_index("ix_bug_reports_task_id", table_name="bug_reports")
    op.drop_index("ix_bug_reports_owner_id", table_name="bug_reports")
    op.drop_table("bug_reports")
