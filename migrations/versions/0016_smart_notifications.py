"""Typed notices and generation-based snooze delivery."""

from alembic import op

revision = "0016_smart_notifications"
down_revision = "0015_custom_structure"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE tasks ADD COLUMN deadline_alert VARCHAR(20) NOT NULL DEFAULT 'default', ADD COLUMN alert_urgent BOOLEAN NOT NULL DEFAULT false"
    )
    op.execute(
        "ALTER TABLE notifications ADD COLUMN category VARCHAR(40) NOT NULL DEFAULT 'reminder', ADD COLUMN dedup_key VARCHAR(200) UNIQUE, ADD COLUMN importance VARCHAR(20) NOT NULL DEFAULT 'normal', ADD COLUMN target JSONB NOT NULL DEFAULT '{}', ADD COLUMN eligible_at TIMESTAMPTZ, ADD COLUMN generation INTEGER NOT NULL DEFAULT 1"
    )
    op.execute("ALTER TABLE delivery_attempts ADD COLUMN generation INTEGER NOT NULL DEFAULT 1")
    op.execute(
        "ALTER TABLE delivery_attempts DROP CONSTRAINT delivery_attempts_notification_id_subscription_id_key"
    )
    op.execute(
        "ALTER TABLE delivery_attempts ADD CONSTRAINT uq_delivery_generation UNIQUE (notification_id, subscription_id, generation)"
    )


def downgrade():
    raise RuntimeError("Restore an isolated backup to downgrade without losing notice history.")
