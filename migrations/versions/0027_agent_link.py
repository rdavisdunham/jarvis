"""Agent link: linked hosts with leases, command journal, captured turns, questions and collector parity."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0027_agent_link"
down_revision = "0026_task_plan"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "agent_hosts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), sa.ForeignKey("user_accounts.id"), nullable=False),
        sa.Column("credential_id", sa.String(36), sa.ForeignKey("bot_credentials.id"), nullable=False),
        sa.Column("server_id", sa.String(100), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("machine", sa.String(100), nullable=False),
        sa.Column("boot_id", sa.String(100), nullable=False),
        sa.Column("proto", sa.Integer(), nullable=False),
        sa.Column("conn_epoch", sa.Integer(), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("caps", postgresql.JSONB(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "server_id", name="uq_agent_host"),
    )
    op.create_index("ix_agent_hosts_owner_id", "agent_hosts", ["owner_id"])
    op.create_index("ix_agent_hosts_account_id", "agent_hosts", ["account_id"])
    op.create_index("ix_agent_hosts_credential_id", "agent_hosts", ["credential_id"])

    op.create_table(
        "agent_commands",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=False),
        sa.Column("op_id", sa.String(100), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=True),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "op_id", name="uq_agent_command_op"),
    )
    op.create_index("ix_agent_commands_owner_id", "agent_commands", ["owner_id"])
    op.create_index("ix_agent_commands_host_id_status", "agent_commands", ["host_id", "status"])

    op.add_column("agent_sessions", sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=True))
    op.add_column("agent_sessions", sa.Column("surf", sa.String(100), nullable=False, server_default=""))
    op.add_column("agent_sessions", sa.Column("incarnation", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("agent_sessions", sa.Column("state", sa.String(20), nullable=False, server_default=""))
    op.add_column("agent_sessions", sa.Column("state_since", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_sessions", sa.Column("state_detail", postgresql.JSONB(), nullable=False, server_default="{}"))
    op.add_column("agent_sessions", sa.Column("last_turn_seq", sa.Integer(), nullable=False, server_default="0"))
    op.create_index("ix_agent_sessions_host_id", "agent_sessions", ["host_id"])

    op.create_table(
        "agent_turns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("session_pk", sa.String(36), sa.ForeignKey("agent_sessions.id"), nullable=False),
        sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=True),
        sa.Column("provider_session_id", sa.String(100), nullable=False),
        sa.Column("provider_turn_id", sa.String(200), nullable=False),
        sa.Column("surf", sa.String(100), nullable=False),
        sa.Column("incarnation", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=True),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_pk", "provider_turn_id", name="uq_agent_turn"),
    )
    op.create_index("ix_agent_turns_owner_id", "agent_turns", ["owner_id"])
    op.create_index("ix_agent_turns_host_id", "agent_turns", ["host_id"])
    op.create_index("ix_agent_turns_session_pk_seq", "agent_turns", ["session_pk", "seq"])

    op.create_table(
        "agent_questions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("session_pk", sa.String(36), sa.ForeignKey("agent_sessions.id"), nullable=False),
        sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=True),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.String(100), nullable=False),
        sa.Column("provider_request_id", sa.String(200), nullable=False),
        sa.Column("group_id", sa.String(100), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("items", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("answers", postgresql.JSONB(), nullable=True),
        sa.Column("answered_via", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_pk", "provider_request_id", name="uq_agent_question"),
    )
    op.create_index("ix_agent_questions_owner_id", "agent_questions", ["owner_id"])
    op.create_index("ix_agent_questions_host_id", "agent_questions", ["host_id"])

    op.create_table(
        "agent_parity",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=True),
        sa.Column("tool", sa.String(30), nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("matched", sa.Boolean(), nullable=False),
        sa.Column("missing", sa.Boolean(), nullable=False),
        sa.Column("diff", postgresql.JSONB(), nullable=False),
        sa.Column("reports", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "tool", "session_id", name="uq_agent_parity"),
    )
    op.create_index("ix_agent_parity_owner_id", "agent_parity", ["owner_id"])
    op.create_index("ix_agent_parity_host_id", "agent_parity", ["host_id"])


def downgrade():
    op.drop_table("agent_parity")
    op.drop_table("agent_questions")
    op.drop_table("agent_turns")
    op.drop_index("ix_agent_sessions_host_id", table_name="agent_sessions")
    for column in ("last_turn_seq", "state_detail", "state_since", "state", "incarnation", "surf", "host_id"):
        op.drop_column("agent_sessions", column)
    op.drop_table("agent_commands")
    op.drop_table("agent_hosts")
