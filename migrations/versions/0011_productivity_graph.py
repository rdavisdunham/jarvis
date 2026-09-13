"""Spaces, outcomes, project relationships and connected notes."""

from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision = "0011_productivity_graph"
down_revision = "0010_unified_planning"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "\nCREATE TABLE actors (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tname VARCHAR(100) NOT NULL, \n\tkind VARCHAR(20) NOT NULL, \n\tarchived BOOLEAN NOT NULL, \n\trevision INTEGER NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT uq_actor_owner_name UNIQUE (owner_id, name)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_actors_owner_id ON actors (owner_id)")
    op.execute(
        "\nCREATE TABLE spaces (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tname VARCHAR(200) NOT NULL, \n\tdescription TEXT NOT NULL, \n\tarchived BOOLEAN NOT NULL, \n\trevision INTEGER NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT uq_space_owner_name UNIQUE (owner_id, name)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_spaces_owner_id ON spaces (owner_id)")
    op.execute(
        "\nCREATE TABLE areas (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tspace_id VARCHAR(36) NOT NULL, \n\tname VARCHAR(200) NOT NULL, \n\tdescription TEXT NOT NULL, \n\tarchived BOOLEAN NOT NULL, \n\trevision INTEGER NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tCONSTRAINT uq_area_space_name UNIQUE (space_id, name), \n\tFOREIGN KEY(space_id) REFERENCES spaces (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_areas_owner_id ON areas (owner_id)")
    op.execute("CREATE INDEX ix_areas_space_id ON areas (space_id)")
    op.execute(
        "\nCREATE TABLE goals (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tspace_id VARCHAR(36), \n\tarea_id VARCHAR(36), \n\tparent_goal_id VARCHAR(36), \n\tname VARCHAR(200) NOT NULL, \n\tdescription TEXT NOT NULL, \n\tsuccess_criteria TEXT NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\thorizon VARCHAR(20) NOT NULL, \n\ttarget_date DATE, \n\tmetric_unit VARCHAR(80) NOT NULL, \n\tmetric_baseline NUMERIC NOT NULL, \n\tmetric_current NUMERIC, \n\tmetric_target NUMERIC, \n\tarchived BOOLEAN NOT NULL, \n\tcompleted_at TIMESTAMP WITH TIME ZONE, \n\trevision INTEGER NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(space_id) REFERENCES spaces (id), \n\tFOREIGN KEY(area_id) REFERENCES areas (id), \n\tFOREIGN KEY(parent_goal_id) REFERENCES goals (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_goals_space_id ON goals (space_id)")
    op.execute("CREATE INDEX ix_goals_parent_goal_id ON goals (parent_goal_id)")
    op.execute("CREATE INDEX ix_goals_owner_id ON goals (owner_id)")
    op.execute("CREATE INDEX ix_goals_area_id ON goals (area_id)")
    op.execute(
        "\nCREATE TABLE goal_project_links (\n\tgoal_id VARCHAR(36) NOT NULL, \n\tproject_id VARCHAR(36) NOT NULL, \n\tPRIMARY KEY (goal_id, project_id), \n\tFOREIGN KEY(goal_id) REFERENCES goals (id), \n\tFOREIGN KEY(project_id) REFERENCES projects (id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE note_goal_links (\n\tnote_id VARCHAR(36) NOT NULL, \n\tgoal_id VARCHAR(36) NOT NULL, \n\tPRIMARY KEY (note_id, goal_id), \n\tFOREIGN KEY(note_id) REFERENCES notes (id), \n\tFOREIGN KEY(goal_id) REFERENCES goals (id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE note_note_links (\n\tnote_id VARCHAR(36) NOT NULL, \n\trelated_note_id VARCHAR(36) NOT NULL, \n\tPRIMARY KEY (note_id, related_note_id), \n\tFOREIGN KEY(note_id) REFERENCES notes (id), \n\tFOREIGN KEY(related_note_id) REFERENCES notes (id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE note_project_links (\n\tnote_id VARCHAR(36) NOT NULL, \n\tproject_id VARCHAR(36) NOT NULL, \n\tPRIMARY KEY (note_id, project_id), \n\tFOREIGN KEY(note_id) REFERENCES notes (id), \n\tFOREIGN KEY(project_id) REFERENCES projects (id)\n)\n\n"
    )
    op.execute("ALTER TABLE projects ADD COLUMN space_id VARCHAR(36) REFERENCES spaces(id)")
    op.execute("CREATE INDEX ix_projects_space_id ON projects (space_id)")
    op.execute("ALTER TABLE projects ADD COLUMN area_id VARCHAR(36) REFERENCES areas(id)")
    op.execute("CREATE INDEX ix_projects_area_id ON projects (area_id)")
    op.execute("ALTER TABLE tasks ADD COLUMN space_id VARCHAR(36) REFERENCES spaces(id)")
    op.execute("CREATE INDEX ix_tasks_space_id ON tasks (space_id)")
    op.execute("ALTER TABLE tasks ADD COLUMN area_id VARCHAR(36) REFERENCES areas(id)")
    op.execute("CREATE INDEX ix_tasks_area_id ON tasks (area_id)")
    op.execute("ALTER TABLE notes ADD COLUMN space_id VARCHAR(36) REFERENCES spaces(id)")
    op.execute("CREATE INDEX ix_notes_space_id ON notes (space_id)")
    op.execute("ALTER TABLE notes ADD COLUMN area_id VARCHAR(36) REFERENCES areas(id)")
    op.execute("CREATE INDEX ix_notes_area_id ON notes (area_id)")
    op.execute("ALTER TABLE projects ADD COLUMN status VARCHAR(30) NOT NULL DEFAULT 'planned'")
    op.execute("ALTER TABLE projects ADD COLUMN success_criteria TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE projects ADD COLUMN start_date DATE")
    op.execute("ALTER TABLE projects ADD COLUMN target_date DATE")
    op.execute("ALTER TABLE projects ADD COLUMN completed_at TIMESTAMPTZ")
    op.execute("ALTER TABLE projects ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT now()")
    op.execute("ALTER TABLE tasks ADD COLUMN planned_date DATE")
    op.execute("ALTER TABLE tasks ADD COLUMN estimate_minutes INTEGER")
    op.execute("ALTER TABLE tasks ADD COLUMN assignee_id VARCHAR(36) REFERENCES actors(id)")
    op.execute("CREATE INDEX ix_tasks_planned_date ON tasks (planned_date)")
    op.execute("CREATE INDEX ix_tasks_assignee_id ON tasks (assignee_id)")
    db = op.get_bind()
    # Keep every existing task, project, note, due date, alert and completion identity.
    # Unclassified work stays unclassified; no guessed Personal/Business assignment.
    actors = list(db.execute(sa.text("SELECT DISTINCT owner_id,assignee FROM tasks")).mappings())
    for row in actors:
        aid = str(uuid4())
        db.execute(
            sa.text(
                "INSERT INTO actors(id,owner_id,name,kind,archived,revision,created_at) VALUES (:id,:owner,:name,:kind,false,1,now())"
            ),
            {
                "id": aid,
                "owner": row["owner_id"],
                "name": row["assignee"],
                "kind": "agent" if row["assignee"].casefold() in {"eri", "agent", "eridani"} else "person",
            },
        )
        db.execute(
            sa.text("UPDATE tasks SET assignee_id=:id WHERE owner_id=:owner AND assignee=:name"),
            {"id": aid, "owner": row["owner_id"], "name": row["assignee"]},
        )
    db.execute(
        sa.text(
            "INSERT INTO note_project_links(note_id,project_id) SELECT id,project_id FROM notes WHERE project_id IS NOT NULL"
        )
    )


def downgrade():
    # Restoring a pre-upgrade backup is required to recover graph data after rollback.
    for table, columns in {
        "tasks": ["assignee_id", "estimate_minutes", "planned_date", "area_id", "space_id"],
        "notes": ["area_id", "space_id"],
        "projects": [
            "status",
            "success_criteria",
            "start_date",
            "target_date",
            "completed_at",
            "updated_at",
            "area_id",
            "space_id",
        ],
    }.items():
        for column in columns:
            op.drop_column(table, column)
    for table in [
        "note_note_links",
        "note_project_links",
        "note_goal_links",
        "goal_project_links",
        "goals",
        "areas",
        "spaces",
        "actors",
    ]:
        op.drop_table(table)
    op.execute("UPDATE owner_settings SET values = values - 'productivity_initialized'")
