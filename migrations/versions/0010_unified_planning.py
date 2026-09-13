"""Unified task alerts, first-party calendar entries and Linear synchronization."""

from uuid import uuid4
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from alembic import op

revision = "0010_unified_planning"
down_revision = "0009_calendar_writes"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE tasks ADD COLUMN is_template BOOLEAN NOT NULL DEFAULT false")
    op.execute("ALTER TABLE tasks ADD COLUMN external JSONB NOT NULL DEFAULT '{}'::jsonb")
    op.execute("ALTER TABLE google_calendars ADD COLUMN details_version INTEGER NOT NULL DEFAULT 0")
    op.execute(
        "\nCREATE TABLE planning_entries (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tkind VARCHAR(20) NOT NULL, \n\ttask_id VARCHAR(36), \n\tfields JSONB NOT NULL, \n\tstatus VARCHAR(20) NOT NULL, \n\trevision INTEGER NOT NULL, \n\tgoogle_calendar_id VARCHAR(36), \n\tgoogle_event_id VARCHAR(1024), \n\tgoogle_snapshot JSONB, \n\tgoogle_job_id VARCHAR(36), \n\tgoogle_state VARCHAR(30) NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(task_id) REFERENCES tasks (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_planning_entries_owner_id ON planning_entries (owner_id)")
    op.execute("CREATE INDEX ix_planning_entries_task_id ON planning_entries (task_id)")
    op.execute(
        "\nCREATE TABLE linear_connections (\n\towner_id VARCHAR(100) NOT NULL, \n\tcredentials TEXT, \n\tworkspace_id VARCHAR(100) NOT NULL, \n\tworkspace_name VARCHAR(250) NOT NULL, \n\tviewer_id VARCHAR(100) NOT NULL, \n\tenabled BOOLEAN NOT NULL, \n\tgeneration INTEGER NOT NULL, \n\trevision INTEGER NOT NULL, \n\tteam_ids JSONB NOT NULL, \n\tonly_mine BOOLEAN NOT NULL, \n\tdirectory JSONB NOT NULL, \n\tlast_sync_at TIMESTAMP WITH TIME ZONE, \n\tfull_sync_at TIMESTAMP WITH TIME ZONE, \n\tnext_sync_at TIMESTAMP WITH TIME ZONE, \n\tstatus VARCHAR(30) NOT NULL, \n\terror VARCHAR(200) NOT NULL, \n\tPRIMARY KEY (owner_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE linear_issues (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tworkspace_id VARCHAR(100) NOT NULL, \n\tremote_id VARCHAR(100) NOT NULL, \n\ttask_id VARCHAR(36) NOT NULL, \n\tsnapshot JSONB NOT NULL, \n\tpending_job_id VARCHAR(36), \n\tsync_state VARCHAR(30) NOT NULL, \n\tlatest_remote JSONB, \n\tPRIMARY KEY (id), \n\tCONSTRAINT uq_linear_issue UNIQUE (owner_id, workspace_id, remote_id), \n\tUNIQUE (task_id), \n\tFOREIGN KEY(task_id) REFERENCES tasks (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_linear_issues_owner_id ON linear_issues (owner_id)")

    db = op.get_bind()
    # Retain schedule/occurrence/notification identities and command receipts.
    # Alert times are never promoted into task deadlines.
    for row in list(db.execute(sa.text("SELECT * FROM schedules WHERE task_id IS NULL")).mappings()):
        tid = str(uuid4())
        status = row["status"] if row["status"] in {"completed", "cancelled"} else "open"
        project = db.execute(
            sa.text("SELECT name FROM projects WHERE id=:id"), {"id": row["project_id"]}
        ).scalar()
        db.execute(
            sa.text("""
            INSERT INTO tasks (id,owner_id,title,notes,status,priority,project,project_id,revision,
                               created_at,updated_at,completed_at,archived,is_template)
            VALUES (:id,:owner,:title,'',:status,0,:project,:pid,1,:created,:created,:completed,false,:template)
        """),
            {
                "id": tid,
                "owner": row["owner_id"],
                "title": row["title"],
                "status": status,
                "project": project,
                "pid": row["project_id"],
                "created": row["created_at"],
                "completed": row["completed_at"] if status == "completed" else None,
                "template": bool(row["recurrence"]),
            },
        )
        db.execute(
            sa.text("UPDATE schedules SET task_id=:tid,kind=:kind WHERE id=:sid"),
            {"tid": tid, "sid": row["id"], "kind": "recurring_task" if row["recurrence"] else "reminder"},
        )
        if row["recurrence"]:
            occurrences = list(
                db.execute(
                    sa.text("""
                SELECT n.id,n.task_id,n.occurrence_id,n.completed_at,n.scheduled_at,n.created_at
                FROM notifications n JOIN schedule_occurrences o ON o.id=n.occurrence_id
                WHERE o.schedule_id=:sid AND n.owner_id=:owner
            """),
                    {"sid": row["id"], "owner": row["owner_id"]},
                ).mappings()
            )
            for notice in occurrences:
                child = (
                    notice["task_id"]
                    or db.execute(
                        sa.text("SELECT id FROM tasks WHERE occurrence_id=:oid AND owner_id=:owner"),
                        {"oid": notice["occurrence_id"], "owner": row["owner_id"]},
                    ).scalar()
                )
                if not child:
                    child = str(uuid4())
                    db.execute(
                        sa.text("""
                        INSERT INTO tasks (id,owner_id,title,notes,status,priority,project,project_id,revision,
                            created_at,updated_at,completed_at,archived,occurrence_id,due_date)
                        VALUES (:id,:owner,:title,'',:status,0,:project,:pid,1,:created,:created,:completed,false,:oid,:due)
                    """),
                        {
                            "id": child,
                            "owner": row["owner_id"],
                            "title": row["title"],
                            "status": "completed" if notice["completed_at"] else "open",
                            "project": project,
                            "pid": row["project_id"],
                            "created": notice["created_at"],
                            "completed": notice["completed_at"],
                            "oid": notice["occurrence_id"],
                            "due": notice["scheduled_at"].astimezone(ZoneInfo(row["timezone"])).date(),
                        },
                    )
                db.execute(
                    sa.text("UPDATE tasks SET parent_task_id=:parent WHERE id=:id"),
                    {"parent": tid, "id": child},
                )
                db.execute(
                    sa.text("UPDATE notifications SET task_id=:tid WHERE id=:id"),
                    {"tid": child, "id": notice["id"]},
                )
        else:
            db.execute(
                sa.text("""
                UPDATE notifications SET task_id=:tid WHERE task_id IS NULL AND occurrence_id IN
                (SELECT id FROM schedule_occurrences WHERE schedule_id=:sid)
            """),
                {"tid": tid, "sid": row["id"]},
            )
    # Re-upgrades retain root tasks created by a previous run/downgrade.
    db.execute(
        sa.text(
            "UPDATE tasks SET is_template=true WHERE id IN (SELECT task_id FROM schedules WHERE kind='recurring_task')"
        )
    )


def downgrade():
    # Keep converted tasks and their links/history. Never delete owner work on rollback.
    op.drop_table("linear_issues")
    op.drop_table("linear_connections")
    op.drop_table("planning_entries")
    op.drop_column("google_calendars", "details_version")
    op.drop_column("tasks", "external")
    op.drop_column("tasks", "is_template")
