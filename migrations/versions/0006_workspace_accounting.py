"""Project organization and durable budget activity/settlement metadata."""

import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006_workspace_accounting"
down_revision = "0005_task_due_time"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "projects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False, index=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("owner_id", "name", name="uq_project_owner_name"),
    )
    op.add_column("tasks", sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id")))
    op.add_column("tasks", sa.Column("parent_task_id", sa.String(36), sa.ForeignKey("tasks.id")))
    op.add_column("tasks", sa.Column("assignee", sa.String(100), nullable=False, server_default="owner"))
    op.add_column("tasks", sa.Column("work_type", sa.String(80), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column("tags", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column("schedules", sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id")))
    for table, column in [("tasks", "project_id"), ("tasks", "parent_task_id"), ("schedules", "project_id")]:
        op.create_index(f"ix_{table}_{column}", table, [column])
    db = op.get_bind()
    for owner, name in db.execute(
        sa.text("SELECT DISTINCT owner_id, project FROM tasks WHERE project IS NOT NULL AND project <> ''")
    ):
        pid = str(uuid.uuid4())
        db.execute(
            sa.text("INSERT INTO projects (id,owner_id,name) VALUES (:id,:owner,:name)"),
            {"id": pid, "owner": owner, "name": name},
        )
        db.execute(
            sa.text("UPDATE tasks SET project_id=:id WHERE owner_id=:owner AND project=:name"),
            {"id": pid, "owner": owner, "name": name},
        )
    op.add_column("budget_reservations", sa.Column("last_activity_at", sa.DateTime(timezone=True)))
    db.execute(sa.text("UPDATE budget_reservations SET last_activity_at=created_at"))
    op.alter_column("budget_reservations", "last_activity_at", nullable=False)
    op.add_column("budget_reservations", sa.Column("closed_at", sa.DateTime(timezone=True)))
    op.add_column(
        "budget_reservations",
        sa.Column("settlement", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )


def downgrade():
    for column in ["settlement", "closed_at", "last_activity_at"]:
        op.drop_column("budget_reservations", column)
    op.drop_column("schedules", "project_id")
    for column in ["tags", "work_type", "assignee", "parent_task_id", "project_id"]:
        op.drop_column("tasks", column)
    op.drop_table("projects")
