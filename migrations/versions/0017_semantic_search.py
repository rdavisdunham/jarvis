"""Derived semantic search and isolated vocabulary evidence."""

from alembic import op

revision = "0017_semantic_search"
down_revision = "0016_smart_notifications"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "\nCREATE TABLE search_documents (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\ttarget_key VARCHAR(300) NOT NULL, \n\tfingerprint VARCHAR(64) NOT NULL, \n\tcontent TEXT NOT NULL, \n\tembedding_model VARCHAR(100) NOT NULL, \n\tvectors JSONB NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, target_key)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_search_documents_owner_id ON search_documents (owner_id)")
    op.execute(
        "\nCREATE TABLE search_index_states (\n\towner_id VARCHAR(100) NOT NULL, \n\tgeneration INTEGER NOT NULL, \n\tindexed_generation INTEGER NOT NULL, \n\tjob_id VARCHAR(36), \n\tstatus VARCHAR(30) NOT NULL, \n\tdocument_count INTEGER NOT NULL, \n\terror VARCHAR(200), \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (owner_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE search_preferences (\n\towner_id VARCHAR(100) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tlearning BOOLEAN NOT NULL, \n\tPRIMARY KEY (owner_id, account_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE search_aliases (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tphrase VARCHAR(200) NOT NULL, \n\ttarget_key VARCHAR(300) NOT NULL, \n\ttarget_fingerprint VARCHAR(64) NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\trevision INTEGER NOT NULL, \n\treview_fingerprint VARCHAR(64), \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, account_id, phrase, target_key)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_search_aliases_owner_id ON search_aliases (owner_id)")
    op.execute("CREATE INDEX ix_search_aliases_account_id ON search_aliases (account_id)")
    op.execute(
        "\nCREATE TABLE search_sessions (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tconversation_id VARCHAR(36), \n\twork_id VARCHAR(36), \n\trequest_key VARCHAR(150) NOT NULL, \n\tquery TEXT NOT NULL, \n\tcandidates JSONB NOT NULL, \n\tresult_ids JSONB NOT NULL, \n\tselected_key VARCHAR(300), \n\tphrase VARCHAR(200), \n\tselected_records JSONB NOT NULL, \n\talias_id VARCHAR(36), \n\toutcome VARCHAR(30) NOT NULL, \n\tsignal VARCHAR(100), \n\tpresented_at TIMESTAMP WITH TIME ZONE, \n\taccepted_at TIMESTAMP WITH TIME ZONE, \n\tsuppressed BOOLEAN NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, account_id, request_key)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_search_sessions_work_id ON search_sessions (work_id)")
    op.execute("CREATE INDEX ix_search_sessions_owner_id ON search_sessions (owner_id)")
    op.execute("CREATE INDEX ix_search_sessions_account_id ON search_sessions (account_id)")
    op.execute("CREATE INDEX ix_search_sessions_conversation_id ON search_sessions (conversation_id)")


def downgrade():
    raise RuntimeError(
        "Disable semantic search to roll back; retain learned vocabulary and restore an isolated backup if needed."
    )
