"""Invited accounts and isolated shared workspace membership."""

from alembic import op

revision = "0012_accounts"
down_revision = "0011_productivity_graph"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "\nCREATE TABLE user_accounts (\n\tid VARCHAR(100) NOT NULL, \n\tname VARCHAR(100) NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE shared_workspaces (\n\tid VARCHAR(36) NOT NULL, \n\tname VARCHAR(200) NOT NULL, \n\tkind VARCHAR(20) NOT NULL, \n\tcreator_id VARCHAR(100) NOT NULL, \n\troot_id VARCHAR(36), \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(creator_id) REFERENCES user_accounts (id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE workspace_members (\n\tworkspace_id VARCHAR(36) NOT NULL, \n\taccount_id VARCHAR(100) NOT NULL, \n\trole VARCHAR(20) NOT NULL, \n\tactive BOOLEAN NOT NULL, \n\tactor_id VARCHAR(36), \n\trevision INTEGER NOT NULL, \n\tPRIMARY KEY (workspace_id, account_id), \n\tFOREIGN KEY(workspace_id) REFERENCES shared_workspaces (id), \n\tFOREIGN KEY(account_id) REFERENCES user_accounts (id), \n\tFOREIGN KEY(actor_id) REFERENCES actors (id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE workspace_invites (\n\tid VARCHAR(36) NOT NULL, \n\tworkspace_id VARCHAR(36), \n\tinviter_id VARCHAR(100) NOT NULL, \n\temail VARCHAR(320) NOT NULL, \n\trole VARCHAR(20) NOT NULL, \n\tstatus VARCHAR(20) NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\taccepted_by VARCHAR(100), \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(workspace_id) REFERENCES shared_workspaces (id), \n\tFOREIGN KEY(inviter_id) REFERENCES user_accounts (id), \n\tFOREIGN KEY(accepted_by) REFERENCES user_accounts (id)\n)\n\n"
    )
    op.execute("CREATE INDEX ix_workspace_invites_email ON workspace_invites (email)")
    op.execute("ALTER TABLE auth_sessions ADD COLUMN workspace_id VARCHAR(36)")
    op.execute("ALTER TABLE commands ADD COLUMN account_id VARCHAR(100)")


def downgrade():
    op.execute("ALTER TABLE commands DROP COLUMN account_id")
    op.execute("ALTER TABLE auth_sessions DROP COLUMN workspace_id")
    op.execute("DROP TABLE workspace_invites")
    op.execute("DROP TABLE workspace_members")
    op.execute("DROP TABLE shared_workspaces")
    op.execute("DROP TABLE user_accounts")
