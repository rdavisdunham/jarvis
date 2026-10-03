"""Record ordering and annotations isolated from synced descriptions."""
from alembic import op
import sqlalchemy as sa
revision = "0019_workspace_details"
down_revision = "0018_note_lists"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("google_event_annotations",
        sa.Column("owner_id",sa.String(100),primary_key=True),
        sa.Column("account_subject",sa.String(255),primary_key=True),
        sa.Column("calendar_id",sa.String(500),primary_key=True),
        sa.Column("event_id",sa.String(500),primary_key=True),
        sa.Column("local_notes",sa.Text(),nullable=False,server_default=""),
        sa.Column("revision",sa.Integer(),nullable=False,server_default="1"))
    op.add_column("planning_entries", sa.Column("local_notes", sa.Text(), nullable=False, server_default=""))
    op.add_column("structure_records", sa.Column("sort_order", sa.Float(), nullable=False, server_default="0"))
    op.add_column("structure_records", sa.Column("local_notes", sa.Text(), nullable=False, server_default=""))
    op.execute("""UPDATE structure_records AS r SET sort_order = s.position * 1024
        FROM (SELECT id, row_number() OVER (PARTITION BY owner_id, parent_id ORDER BY created_at, id) AS position
        FROM structure_records) AS s WHERE r.id = s.id""")

def downgrade():
    op.drop_column("planning_entries", "local_notes")
    op.drop_table("google_event_annotations")
    op.drop_column("structure_records", "local_notes")
    op.drop_column("structure_records", "sort_order")
