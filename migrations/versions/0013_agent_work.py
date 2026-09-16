"""Durable requests, temporary device bridges and action receipts."""

from alembic import op

revision = "0013_agent_work"
down_revision = "0012_accounts"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE google_oauth_attempts ADD COLUMN return_to VARCHAR(1000)")
    op.execute(
        "CREATE TABLE agent_work (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tdevice_id VARCHAR(36) NOT NULL, \n\tconversation_id VARCHAR(36) NOT NULL, \n\tparent_id VARCHAR(36), \n\tvoice_session_id VARCHAR(36), \n\tinput_hash VARCHAR(64) NOT NULL, \n\tinput_ciphertext TEXT, \n\tcheckpoint_ciphertext TEXT, \n\trevision INTEGER NOT NULL, \n\tcancel_requested BOOLEAN NOT NULL, \n\tdependencies JSONB NOT NULL, \n\tresources JSONB NOT NULL, \n\tresult JSONB NOT NULL, \n\ttransient BOOLEAN NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tseen_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(id) REFERENCES jobs (id), \n\tFOREIGN KEY(conversation_id) REFERENCES conversations (id)\n)"
    )
    op.execute("CREATE INDEX ix_agent_work_account_id ON agent_work (account_id)")
    op.execute("CREATE INDEX ix_agent_work_conversation_id ON agent_work (conversation_id)")
    op.execute("CREATE INDEX ix_agent_work_expires_at ON agent_work (expires_at)")
    op.execute("CREATE INDEX ix_agent_work_owner_id ON agent_work (owner_id)")
    op.execute("CREATE INDEX ix_agent_work_parent_id ON agent_work (parent_id)")
    op.execute("CREATE INDEX ix_agent_work_voice_session_id ON agent_work (voice_session_id)")
    op.execute(
        "CREATE TABLE voice_inboxes (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tdevice_id VARCHAR(36) NOT NULL, \n\tconversation_id VARCHAR(36) NOT NULL, \n\tcontent_ciphertext TEXT NOT NULL, \n\tcursor INTEGER NOT NULL, \n\trevision INTEGER NOT NULL, \n\tlast_input_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tend_requested BOOLEAN NOT NULL, \n\tclosed BOOLEAN NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(conversation_id) REFERENCES conversations (id)\n)"
    )
    op.execute("CREATE INDEX ix_voice_inboxes_expires_at ON voice_inboxes (expires_at)")
    op.execute("CREATE INDEX ix_voice_inboxes_owner_id ON voice_inboxes (owner_id)")
    op.execute(
        "CREATE TABLE device_bridges (\n\towner_id VARCHAR(100) NOT NULL, \n\tdevice_id VARCHAR(36) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tcontext_ciphertext TEXT NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (owner_id, device_id)\n)"
    )
    op.execute("CREATE INDEX ix_device_bridges_expires_at ON device_bridges (expires_at)")
    op.execute(
        "CREATE TABLE device_actions (\n\towner_id VARCHAR(100) NOT NULL, \n\tid VARCHAR(100) NOT NULL, \n\tdevice_id VARCHAR(36) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\taction_ciphertext TEXT NOT NULL, \n\tresult_ciphertext TEXT, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tsent_at TIMESTAMP WITH TIME ZONE, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (owner_id, id)\n)"
    )
    op.execute("CREATE INDEX ix_device_actions_device_id ON device_actions (device_id)")
    op.execute("CREATE INDEX ix_device_actions_expires_at ON device_actions (expires_at)")
    op.execute(
        "CREATE TABLE action_changes (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\tcommand_id VARCHAR(100) NOT NULL, \n\ttool VARCHAR(70) NOT NULL, \n\tentity_kind VARCHAR(40) NOT NULL, \n\tentity_id VARCHAR(100) NOT NULL, \n\tbefore_ciphertext TEXT, \n\tafter_ciphertext TEXT, \n\treverted_by VARCHAR(100), \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)"
    )
    op.execute("CREATE INDEX ix_action_changes_account_id ON action_changes (account_id)")
    op.execute("CREATE INDEX ix_action_changes_command_id ON action_changes (command_id)")
    op.execute("CREATE INDEX ix_action_changes_entity_id ON action_changes (entity_id)")
    op.execute("CREATE INDEX ix_action_changes_owner_id ON action_changes (owner_id)")


def downgrade():
    op.execute("DROP TABLE action_changes")
    op.execute("DROP TABLE device_actions")
    op.execute("DROP TABLE device_bridges")
    op.execute("DROP TABLE voice_inboxes")
    op.execute("DROP TABLE agent_work")
    op.execute("ALTER TABLE google_oauth_attempts DROP COLUMN return_to")
