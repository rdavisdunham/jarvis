"""Run freshness: the base a run started from, the remotea version, and migration numbers Eridani reserves."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0033_run_freshness"
down_revision = "0032_plan_review"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_hosts", sa.Column("version", sa.String(40), nullable=True))
    op.add_column("agent_runs", sa.Column("base_facts", JSONB(), nullable=False, server_default="{}"))
    op.add_column("agent_run_plans", sa.Column("migration", JSONB(), nullable=True))
    op.create_table(
        "migration_counters",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False, index=True),
        sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=False, index=True),
        sa.Column("host_target", sa.String(64), nullable=False),
        sa.Column("dir", sa.String(200), nullable=True),
        sa.Column("pattern", sa.String(200), nullable=True),
        sa.Column("override", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "host_id", "host_target", name="uq_migration_counter"),
    )
    op.create_table(
        "migration_reservations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False, index=True),
        sa.Column("counter_id", sa.String(36), sa.ForeignKey("migration_counters.id"), nullable=False, index=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id"), nullable=False, index=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False, index=True),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="held"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("uq_migration_reservation_held", "migration_reservations", ["counter_id", "number"], unique=True,
                    postgresql_where=sa.text("status = 'held'"))


def downgrade():
    op.drop_index("uq_migration_reservation_held", table_name="migration_reservations")
    op.drop_table("migration_reservations")
    op.drop_table("migration_counters")
    op.drop_column("agent_run_plans", "migration")
    op.drop_column("agent_runs", "base_facts")
    op.drop_column("agent_hosts", "version")
