"""Linear-mapping placements become routing observations (organization knowledge, phase 2).

Backfills one `linear_mapping` observation for every active record the Linear mapping filed and that still sits in
the Home the mapping gave it (`provenance.linear_home`). New placements are recorded by `routing.observe_linear`.
These observations feed overviews and later placement by meaning; phrase rules read human observations only."""
from alembic import op

revision = "0053_linear_placement_evidence"
down_revision = "0052_record_overviews"
branch_labels = depends_on = None


BACKFILL = """
        INSERT INTO routing_observations
            (id, owner_id, record_id, source_key, record_revision, schema_revision, origin, evidence, suppressed,
             created_at)
        SELECT gen_random_uuid()::text, r.owner_id, r.id, 'linear-mapping:' || r.id, r.revision,
               COALESCE(s.revision, 1), 'linear_mapping',
               jsonb_build_object(
                   'title', r.title, 'type_id', r.type_id,
                   'assignment', jsonb_build_object('parent_id', r.parent_id, 'values', '{}'::jsonb),
                   'provenance', jsonb_build_object(
                       'kind', 'linear_mapping', 'identifier', l.snapshot->>'identifier',
                       'team', l.snapshot->'team'->>'name', 'project', l.snapshot->'project'->>'name')),
               false, COALESCE(r.updated_at, now())
        FROM structure_records r
        JOIN linear_issues l ON l.task_id = r.task_id AND l.owner_id = r.owner_id
        LEFT JOIN structure_schemas s ON s.owner_id = r.owner_id
        WHERE r.archived = false AND r.parent_id IS NOT NULL AND r.provenance->>'linear_home' = r.parent_id
        ON CONFLICT (owner_id, source_key) DO NOTHING
"""


def upgrade():
    op.execute(BACKFILL)


def downgrade():
    op.execute("DELETE FROM routing_observations WHERE origin = 'linear_mapping'")
