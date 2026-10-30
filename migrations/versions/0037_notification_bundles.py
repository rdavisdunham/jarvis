"""Smart notifications: distinct snoozed and delivered states, and bundles that carry several notices."""

from alembic import op

revision = "0037_notification_bundles"
down_revision = "0036_run_summaries"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE notifications ADD COLUMN snoozed_until TIMESTAMPTZ, ADD COLUMN delivered_at TIMESTAMPTZ, "
        "ADD COLUMN bundle_id VARCHAR(36) REFERENCES notifications(id)"
    )
    op.execute("CREATE INDEX ix_notifications_bundle_id ON notifications (bundle_id)")
    # Notices already pushed keep that fact; the previous release only recorded it on delivery rows.
    op.execute(
        "UPDATE notifications n SET delivered_at = d.at FROM (SELECT notification_id, generation, "
        "max(next_attempt_at) AS at FROM delivery_attempts WHERE status = 'submitted' "
        "GROUP BY notification_id, generation) d WHERE d.notification_id = n.id AND d.generation = n.generation"
    )


def downgrade():
    op.execute("DROP INDEX ix_notifications_bundle_id")
    op.execute(
        "ALTER TABLE notifications DROP COLUMN bundle_id, DROP COLUMN delivered_at, DROP COLUMN snoozed_until"
    )
