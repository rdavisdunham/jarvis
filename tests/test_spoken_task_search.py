import pytest
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import Task
from jarvis.task_context import resolve
from jarvis.task_tools import list_tasks


@pytest.mark.parametrize(
    ("stored", "spoken"),
    [("test, test, one, two, three", "test test 123"), ("test test 123", "test, test, one, two, three"), ("Route 66 permit", "route sixty-six")],
)
def test_task_search_matches_spoken_numbers(stored, spoken):
    owner = get_settings().owner_id
    with session_scope() as db:
        db.add(Task(owner_id=owner, title=stored))
        db.add(Task(owner_id=owner, title="Unrelated errand"))
    with session_scope() as db:
        assert [t["title"] for t in list_tasks(db, owner, {"query": spoken})["tasks"]] == [stored]
        assert [t["title"] for t in resolve(db, owner, {}, None, "search", spoken)["tasks"]] == [stored]
