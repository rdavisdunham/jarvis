"""Hyperplan: run roles (planner, reviewer, worker), the review counter and the plan review rounds (agent_plan_reviews)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0041_hyperplan"
down_revision = "0040_agent_swarms"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_runs", sa.Column("roles", postgresql.JSONB(), nullable=True))
    op.add_column("agent_runs", sa.Column("plan_round", sa.Integer(), nullable=False, server_default="0"))
    op.create_table(
        "agent_plan_reviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("plan_revision", sa.Integer(), nullable=False),
        sa.Column("harness", sa.String(30), nullable=False),
        sa.Column("model", sa.String(80), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("surf", sa.Text(), nullable=True),
        sa.Column("incarnation", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("session_pk", sa.String(36), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("verdict", sa.String(20), nullable=True),
        sa.Column("via", sa.String(20), nullable=False, server_default=""),
        sa.Column("findings", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("questions", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("resolutions", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("token_hash", sa.String(64), nullable=True, unique=True),
        sa.Column("token_salt", sa.String(64), nullable=False, server_default=""),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("blocked_since", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "round", name="uq_agent_plan_review_round"),
    )
    op.create_index("ix_agent_plan_reviews_owner_id", "agent_plan_reviews", ["owner_id"])
    op.create_index("ix_agent_plan_reviews_run_id", "agent_plan_reviews", ["run_id"])


def downgrade():
    op.drop_index("ix_agent_plan_reviews_run_id", table_name="agent_plan_reviews")
    op.drop_index("ix_agent_plan_reviews_owner_id", table_name="agent_plan_reviews")
    op.drop_table("agent_plan_reviews")
    op.drop_column("agent_runs", "plan_round")
    op.drop_column("agent_runs", "roles")
