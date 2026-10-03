"""Quick capture uses existing tasks; setup progress is account-scoped settings."""
from alembic import op
import sqlalchemy as sa
revision = "0020_capture_setup"
down_revision = "0019_workspace_details"
branch_labels = depends_on = None

def upgrade():
    op.add_column("tasks", sa.Column("is_quick_list",sa.Boolean(),nullable=False,server_default=sa.text("false")))
    op.add_column("tasks", sa.Column("quick_section",sa.String(120),nullable=False,server_default=""))
    op.add_column("tasks", sa.Column("quick_order",sa.Integer(),nullable=False,server_default="0"))

def downgrade():
    for col in ("quick_order","quick_section","is_quick_list"):
        op.drop_column("tasks",col)
