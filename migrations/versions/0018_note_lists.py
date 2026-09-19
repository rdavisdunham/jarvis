"""Saved note lists and source-linked organization."""

from alembic import op

revision = "0018_note_lists"
down_revision = "0017_semantic_search"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "\nCREATE TABLE note_lists (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tname VARCHAR(80) NOT NULL, \n\tdescription TEXT NOT NULL, \n\tfilters JSONB NOT NULL, \n\tautomatic BOOLEAN NOT NULL, \n\textract_entries BOOLEAN NOT NULL, \n\tarchived BOOLEAN NOT NULL, \n\trevision INTEGER NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_note_lists_owner_id ON note_lists (owner_id)")
    op.execute(
        "\nCREATE TABLE note_organizations (\n\tnote_id VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\ttags_locked BOOLEAN NOT NULL, \n\tgenerated BOOLEAN NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tfingerprint VARCHAR(64) NOT NULL, \n\tresult JSONB NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (note_id), \n\tFOREIGN KEY(note_id) REFERENCES notes (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_note_organizations_owner_id ON note_organizations (owner_id)")
    op.execute(
        "\nCREATE TABLE note_entry_sources (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tsource_id VARCHAR(36) NOT NULL, \n\tentry_id VARCHAR(36) NOT NULL, \n\tentry_key VARCHAR(64) NOT NULL, \n\tevidence TEXT NOT NULL, \n\tsource_revision INTEGER NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, source_id, entry_key), \n\tFOREIGN KEY(source_id) REFERENCES notes (id), \n\tFOREIGN KEY(entry_id) REFERENCES notes (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_note_entry_sources_entry_id ON note_entry_sources (entry_id)")
    op.execute("CREATE INDEX ix_note_entry_sources_owner_id ON note_entry_sources (owner_id)")
    op.execute("CREATE INDEX ix_note_entry_sources_source_id ON note_entry_sources (source_id)")


def downgrade():
    op.drop_table("note_entry_sources")
    op.drop_table("note_organizations")
    op.drop_table("note_lists")
