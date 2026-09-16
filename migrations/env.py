from alembic import context
from jarvis.config import get_settings
from jarvis.models import Base
from sqlalchemy import create_engine, text

config = context.config
if context.is_offline_mode():
    context.configure(url=get_settings().database_url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    with create_engine(get_settings().database_url).begin() as connection:
        # API and worker pre-deploy hooks can overlap. Serialize the whole migration.
        connection.execute(text("SELECT pg_advisory_xact_lock(hashtext('eridani:schema-migration'))"))
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
