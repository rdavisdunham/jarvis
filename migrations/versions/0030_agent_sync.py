"""Stage E: per-plan completion policy, auto-link evidence revision and daily collector parity."""
import sqlalchemy as sa
from alembic import op

revision = "0030_agent_sync"
down_revision = "0029_agent_loop"
branch_labels = depends_on = None


def upgrade():
    # What "done" means for a plan: code_ready, pr_open or pr_merged (docs/AGENT_CONTROL.md).
    op.add_column("tasks", sa.Column("completion_policy", sa.String(20), nullable=False, server_default="code_ready"))
    op.add_column("agent_run_plans", sa.Column("completion_policy", sa.String(20), nullable=False,
                                               server_default="code_ready"))
    # The evidence revision an automatic (dream) link was made at; resolve() keeps the link until it changes.
    op.add_column("agent_sessions", sa.Column("binding_evidence", sa.String(64), nullable=False, server_default=""))
    op.create_table(
        "agent_parity_days",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("reports", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("matched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("mismatched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("missing", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "day", name="uq_agent_parity_day"),
    )
    op.create_index("ix_agent_parity_days_owner_id", "agent_parity_days", ["owner_id"])


def downgrade():
    op.drop_table("agent_parity_days")
    op.drop_column("agent_sessions", "binding_evidence")
    op.drop_column("agent_run_plans", "completion_policy")
    op.drop_column("tasks", "completion_policy")
