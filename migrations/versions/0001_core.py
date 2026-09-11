"""Initial durable Jarvis schema, frozen at release 1."""

from alembic import op

revision = "0001_core"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "CREATE TABLE auth_sessions (\n\ttoken_hash VARCHAR(64) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tdevice_id VARCHAR(36) NOT NULL, \n\tcsrf VARCHAR(64) NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (token_hash)\n)"
    )
    op.execute("CREATE INDEX ix_auth_sessions_owner_id ON auth_sessions (owner_id)")
    op.execute(
        "CREATE TABLE budget_reservations (\n\tid VARCHAR(100) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tamount NUMERIC(12, 6) NOT NULL, \n\tactual NUMERIC(12, 6), \n\tmodel VARCHAR(100) NOT NULL, \n\tstate VARCHAR(30) NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)"
    )
    op.execute("CREATE INDEX ix_budget_reservations_owner_id ON budget_reservations (owner_id)")
    op.execute(
        "CREATE TABLE commands (\n\towner_id VARCHAR(100) NOT NULL, \n\tid VARCHAR(100) NOT NULL, \n\trequest_hash VARCHAR(64) NOT NULL, \n\tresult JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (owner_id, id)\n)"
    )
    op.execute(
        "CREATE TABLE conversations (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tdevice_id VARCHAR(36) NOT NULL, \n\tprivate BOOLEAN NOT NULL, \n\tlearning BOOLEAN NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)"
    )
    op.execute("CREATE INDEX ix_conversations_owner_id ON conversations (owner_id)")
    op.execute(
        "CREATE TABLE events (\n\tid SERIAL NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tkind VARCHAR(60) NOT NULL, \n\tentity_id VARCHAR(100) NOT NULL, \n\trevision INTEGER, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)"
    )
    op.execute("CREATE INDEX ix_events_owner_id ON events (owner_id)")
    op.execute(
        "CREATE TABLE jobs (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tkind VARCHAR(50) NOT NULL, \n\tpayload JSONB NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tresult JSONB, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tfinished_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (id)\n)"
    )
    op.execute("CREATE INDEX ix_jobs_owner_id ON jobs (owner_id)")
    op.execute(
        "CREATE TABLE owner_settings (\n\towner_id VARCHAR(100) NOT NULL, \n\tvalues JSONB NOT NULL, \n\tPRIMARY KEY (owner_id)\n)"
    )
    op.execute(
        "CREATE TABLE push_subscriptions (\n\tid VARCHAR(64) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tdevice_id VARCHAR(36) NOT NULL, \n\tsubscription JSONB NOT NULL, \n\tactive BOOLEAN NOT NULL, \n\tPRIMARY KEY (id)\n)"
    )
    op.execute(
        "CREATE TABLE tasks (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\ttitle VARCHAR(500) NOT NULL, \n\tnotes TEXT NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tpriority INTEGER NOT NULL, \n\tproject VARCHAR(200), \n\tdue_date DATE, \n\trevision INTEGER NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tcompleted_at TIMESTAMP WITH TIME ZONE, \n\tarchived BOOLEAN NOT NULL, \n\toccurrence_id VARCHAR(36), \n\tPRIMARY KEY (id), \n\tUNIQUE (occurrence_id)\n)"
    )
    op.execute("CREATE INDEX ix_tasks_owner_id ON tasks (owner_id)")
    op.execute(
        "CREATE TABLE worker_health (\n\tid VARCHAR(30) NOT NULL, \n\tlast_scan_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)"
    )
    op.execute(
        "CREATE TABLE schedules (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\ttitle VARCHAR(500) NOT NULL, \n\ttask_id VARCHAR(36), \n\ttimezone VARCHAR(100) NOT NULL, \n\trecurrence VARCHAR(250), \n\tanchor_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tnext_run_at TIMESTAMP WITH TIME ZONE, \n\tkind VARCHAR(30) NOT NULL, \n\tstatus VARCHAR(20) NOT NULL, \n\trevision INTEGER NOT NULL, \n\toriginal_words TEXT NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(task_id) REFERENCES tasks (id)\n)"
    )
    op.execute("CREATE INDEX ix_schedules_next_run_at ON schedules (next_run_at)")
    op.execute("CREATE INDEX ix_schedules_owner_id ON schedules (owner_id)")
    op.execute(
        "CREATE TABLE sources (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tconversation_id VARCHAR(36), \n\tnative_id VARCHAR(200) NOT NULL, \n\tkind VARCHAR(50) NOT NULL, \n\trole VARCHAR(30) NOT NULL, \n\tcontent TEXT NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tdeleted_at TIMESTAMP WITH TIME ZONE, \n\texplicit BOOLEAN NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, native_id), \n\tFOREIGN KEY(conversation_id) REFERENCES conversations (id)\n)"
    )
    op.execute("CREATE INDEX ix_sources_owner_id ON sources (owner_id)")
    op.execute("CREATE INDEX sources_search ON sources (owner_id, created_at)")
    op.execute(
        "CREATE TABLE usage_events (\n\trequest_id VARCHAR(150) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\treservation_id VARCHAR(100) NOT NULL, \n\tmodel VARCHAR(100) NOT NULL, \n\tamount NUMERIC(12, 6) NOT NULL, \n\ttokens JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (request_id), \n\tFOREIGN KEY(reservation_id) REFERENCES budget_reservations (id)\n)"
    )
    op.execute(
        "CREATE TABLE workflow_outbox (\n\tjob_id VARCHAR(36) NOT NULL, \n\tsubmitted_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (job_id), \n\tFOREIGN KEY(job_id) REFERENCES jobs (id)\n)"
    )
    op.execute(
        "CREATE TABLE memory_assertions (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\tsource_id VARCHAR(36) NOT NULL, \n\tcontent TEXT NOT NULL, \n\tattribution VARCHAR(40) NOT NULL, \n\trevision INTEGER NOT NULL, \n\tsuppressed BOOLEAN NOT NULL, \n\tsupersedes_id VARCHAR(36), \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(source_id) REFERENCES sources (id), \n\tFOREIGN KEY(supersedes_id) REFERENCES memory_assertions (id)\n)"
    )
    op.execute("CREATE INDEX ix_memory_assertions_owner_id ON memory_assertions (owner_id)")
    op.execute(
        "CREATE TABLE schedule_occurrences (\n\tid VARCHAR(36) NOT NULL, \n\tschedule_id VARCHAR(36) NOT NULL, \n\trevision INTEGER NOT NULL, \n\tscheduled_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (schedule_id, revision, scheduled_at), \n\tFOREIGN KEY(schedule_id) REFERENCES schedules (id)\n)"
    )
    op.execute(
        "CREATE TABLE notifications (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(100) NOT NULL, \n\toccurrence_id VARCHAR(36), \n\ttitle VARCHAR(500) NOT NULL, \n\tbody TEXT NOT NULL, \n\ttask_id VARCHAR(36), \n\tscheduled_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tread_at TIMESTAMP WITH TIME ZONE, \n\tdismissed_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (id), \n\tUNIQUE (occurrence_id), \n\tFOREIGN KEY(occurrence_id) REFERENCES schedule_occurrences (id), \n\tFOREIGN KEY(task_id) REFERENCES tasks (id)\n)"
    )
    op.execute("CREATE INDEX ix_notifications_owner_id ON notifications (owner_id)")
    op.execute(
        "CREATE TABLE delivery_attempts (\n\tid VARCHAR(36) NOT NULL, \n\tnotification_id VARCHAR(36) NOT NULL, \n\tsubscription_id VARCHAR(64) NOT NULL, \n\tstatus VARCHAR(30) NOT NULL, \n\tattempts INTEGER NOT NULL, \n\tnext_attempt_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tlease_until TIMESTAMP WITH TIME ZONE, \n\terror_code VARCHAR(100), \n\tPRIMARY KEY (id), \n\tUNIQUE (notification_id, subscription_id), \n\tFOREIGN KEY(notification_id) REFERENCES notifications (id), \n\tFOREIGN KEY(subscription_id) REFERENCES push_subscriptions (id)\n)"
    )


def downgrade():
    raise RuntimeError("Destructive downgrade is disabled. Restore an encrypted backup to a new database.")
