"""Freestyle runs: the operator talks to the agent directly; Eridani records turns but does not judge or cap."""
import sqlalchemy as sa
from alembic import op

revision = "0034_run_freestyle"
down_revision = "0033_run_freshness"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_runs", sa.Column("freestyle", sa.Boolean(), nullable=False, server_default="false"))


def downgrade():
    op.drop_column("agent_runs", "freestyle")
