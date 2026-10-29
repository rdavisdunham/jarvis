"""Plan review: an agent's proposed changes to human-authored criteria, verify or policy, stored with the frozen plan."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0032_plan_review"
down_revision = "0031_agent_security"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_run_plans", sa.Column("proposals", JSONB(), nullable=True))


def downgrade():
    op.drop_column("agent_run_plans", "proposals")
