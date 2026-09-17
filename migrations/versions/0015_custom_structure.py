"""Configurable organization, field understanding and routing evidence."""

from alembic import op

revision = "0015_custom_structure"
down_revision = "0014_external_agents"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "\nCREATE TABLE structure_schemas (\n\towner_id VARCHAR(100) NOT NULL, \n\trevision INTEGER NOT NULL, \n\tdefinition JSONB NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (owner_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE structure_proposals (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tschema_revision INTEGER NOT NULL, \n\tdefinition JSONB NOT NULL, \n\timpact JSONB NOT NULL, \n\tapplied_at TIMESTAMP WITH TIME ZONE, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_structure_proposals_owner_id ON structure_proposals (owner_id)")
    op.execute(
        "\nCREATE TABLE structure_records (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\ttype_id VARCHAR(80) NOT NULL, \n\ttitle VARCHAR(500) NOT NULL, \n\tbody TEXT NOT NULL, \n\tvalues JSONB NOT NULL, \n\tstatus_id VARCHAR(80), \n\tparent_id VARCHAR(36), \n\ttask_id VARCHAR(36), \n\tnote_id VARCHAR(36), \n\tlegacy_kind VARCHAR(30), \n\tarchived BOOLEAN NOT NULL, \n\trevision INTEGER NOT NULL, \n\tschema_revision INTEGER NOT NULL, \n\tprovenance JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(parent_id) REFERENCES structure_records (id), \n\tUNIQUE (task_id), \n\tFOREIGN KEY(task_id) REFERENCES tasks (id), \n\tUNIQUE (note_id), \n\tFOREIGN KEY(note_id) REFERENCES notes (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_structure_records_owner_id ON structure_records (owner_id)")
    op.execute("CREATE INDEX ix_structure_records_parent_id ON structure_records (parent_id)")
    op.execute("CREATE INDEX ix_structure_records_type_id ON structure_records (type_id)")
    op.execute(
        "\nCREATE TABLE structure_links (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\trelationship_id VARCHAR(80) NOT NULL, \n\tsource_id VARCHAR(36) NOT NULL, \n\ttarget_id VARCHAR(36) NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, relationship_id, source_id, target_id), \n\tFOREIGN KEY(source_id) REFERENCES structure_records (id), \n\tFOREIGN KEY(target_id) REFERENCES structure_records (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_structure_links_owner_id ON structure_links (owner_id)")
    op.execute("CREATE INDEX ix_structure_links_source_id ON structure_links (source_id)")
    op.execute("CREATE INDEX ix_structure_links_target_id ON structure_links (target_id)")
    op.execute(
        "\nCREATE TABLE field_understandings (\n\towner_id VARCHAR(100) NOT NULL, \n\tdefinition_id VARCHAR(180) NOT NULL, \n\tfingerprint VARCHAR(64) NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tunderstanding JSONB NOT NULL, \n\tquestions JSONB NOT NULL, \n\tanswers JSONB NOT NULL, \n\trevision INTEGER NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (owner_id, definition_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE routing_observations (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\trecord_id VARCHAR(36) NOT NULL, \n\tsource_key VARCHAR(160) NOT NULL, \n\trecord_revision INTEGER NOT NULL, \n\tschema_revision INTEGER NOT NULL, \n\torigin VARCHAR(30) NOT NULL, \n\tevidence JSONB NOT NULL, \n\tsuppressed BOOLEAN NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, source_key), \n\tFOREIGN KEY(record_id) REFERENCES structure_records (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_routing_observations_owner_id ON routing_observations (owner_id)")
    op.execute("CREATE INDEX ix_routing_observations_record_id ON routing_observations (record_id)")
    op.execute(
        "\nCREATE TABLE routing_patterns (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tfingerprint VARCHAR(64) NOT NULL, \n\tcondition JSONB NOT NULL, \n\tassignment JSONB NOT NULL, \n\tevidence_ids JSONB NOT NULL, \n\treason TEXT NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tschema_revision INTEGER NOT NULL, \n\trevision INTEGER NOT NULL, \n\torigin VARCHAR(30) NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, fingerprint)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_routing_patterns_owner_id ON routing_patterns (owner_id)")
    op.execute(
        "\nCREATE TABLE routing_reviews (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tperiod VARCHAR(80) NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tsummary JSONB NOT NULL, \n\tquestions JSONB NOT NULL, \n\tanswers JSONB NOT NULL, \n\trevision INTEGER NOT NULL, \n\toffered_on VARCHAR(10), \n\tdeferred_until TIMESTAMP WITH TIME ZONE, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tfinished_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, period)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_routing_reviews_owner_id ON routing_reviews (owner_id)")


def downgrade():
    raise RuntimeError(
        "This migration preserves user structure. Restore an isolated backup instead of deleting schema data."
    )
