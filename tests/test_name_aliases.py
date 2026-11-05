"""The owner's "Also known as" names (Settings > Profile, `settings.update` `name_aliases`): stored as alias
identities on their own actor and consulted wherever the app resolves a person by name. Exact matches only
(case-insensitive, whitespace-normalised); a bare first name another actor shares identifies nobody; a name
another actor already has is refused."""

from uuid import uuid4

from jarvis.assignees import alias_matches, resolve_assignee
from jarvis.context import identity
from jarvis.context.models import ContextActorIdentity
from jarvis.db import session_scope
from jarvis.models import Actor, GoogleIdentity
from sqlalchemy import select


def send(client, tool, **args):
    return client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": args})


def cmd(client, tool, **args):
    r = send(client, tool, **args)
    assert r.status_code == 200, r.text
    return r.json()["data"]


def error(client, tool, **args):
    r = send(client, tool, **args)
    assert r.status_code >= 400, r.text
    return r.status_code, r.json()["error"]


def _owner():
    with session_scope() as db:
        return db.scalar(select(Actor.owner_id).where(Actor.name == "owner").limit(1))


def me(client):
    cmd(client, "task.create", title="seed the workspace")  # productivity.defaults creates the owner actor
    with session_scope() as db:
        owner = _owner()
        return owner, identity.owner_actor(db, owner)


def identities(owner, actor, namespace="alias"):
    with session_scope() as db:
        return sorted((r.value, r.status, (r.provenance or {}).get("origin")) for r in db.scalars(
            select(ContextActorIdentity).where(ContextActorIdentity.owner_id == owner,
                                               ContextActorIdentity.actor_id == actor,
                                               ContextActorIdentity.namespace == namespace)))


def test_saving_aliases_replaces_identities_and_seeds_the_sign_in_email(client):
    owner, actor = me(client)
    with session_scope() as db:
        db.merge(GoogleIdentity(owner_id=owner, subject="subject-owner", email="Davis.Dunham@example.test"))
    saved = cmd(client, "settings.update", name_aliases=["  Robert   Davis Dunham ", "Davis Dunham", "davis dunham"])
    assert saved["name_aliases"] == ["Robert Davis Dunham", "Davis Dunham"]
    assert identities(owner, actor) == [("Davis Dunham", "confirmed", "profile"),
                                        ("Robert Davis Dunham", "confirmed", "profile")]
    assert [v for v, *_ in identities(owner, actor, "email")] == ["Davis.Dunham@example.test"]
    with session_scope() as db:  # an alias meeting seeding learned stays when the profile list changes
        identity.add_identity(db, owner, actor, "alias", "DD")
    again = cmd(client, "settings.update", name_aliases=["Davis Dunham"])
    assert again["name_aliases"] == ["Davis Dunham"]
    assert [v for v, *_ in identities(owner, actor)] == ["DD", "Davis Dunham"]
    assert client.get("/api/v1/bootstrap").json()["preferences"]["name_aliases"] == ["Davis Dunham"]
    cmd(client, "settings.update", name_aliases=[])
    assert [v for v, *_ in identities(owner, actor)] == ["DD"]
    assert len(identities(owner, actor, "email")) == 1  # not added twice


def test_invalid_and_taken_aliases_are_refused(client):
    owner, actor = me(client)
    priya = cmd(client, "actor.create", name="Priya Shah")
    assert error(client, "settings.update", name_aliases=["priya  shah"])[1]["code"] == "ALIAS_TAKEN"
    with session_scope() as db:
        identity.add_identity(db, owner, priya["id"], "alias", "PS")
    taken = error(client, "settings.update", name_aliases=["ps"])[1]
    assert taken["code"] == "ALIAS_TAKEN" and "Priya Shah" in taken["message"]
    first = error(client, "settings.update", name_aliases=["Priya"])[1]  # Priya Shah's first name
    assert first["code"] == "ALIAS_TAKEN" and "fuller name" in first["message"]
    for bad in (["me"], ["davis@example.test"], ["x" * 81], [f"Name {i}" for i in range(11)]):
        assert error(client, "settings.update", name_aliases=bad)[0] in (400, 422)
    assert identities(owner, actor) == []


def test_task_assignment_and_filters_resolve_aliases(client):
    owner, actor = me(client)
    cmd(client, "settings.update", name_aliases=["Davis Dunham", "Davis"])
    task = cmd(client, "task.create", title="Rubric update", assignee="  davis   DUNHAM ")
    assert task["assignee_id"] == actor and task["assignee"] == "owner"
    assert cmd(client, "task.create", title="Short", assignee="Davis")["assignee_id"] == actor
    with session_scope() as db:
        assert resolve_assignee(db, owner, "Davis Dunham") == actor
        assert db.scalar(select(Actor.id).where(Actor.owner_id == owner, Actor.name == "Davis Dunham")) is None


def test_a_bare_first_name_another_actor_shares_matches_nobody(client):
    owner, actor = me(client)
    cmd(client, "settings.update", name_aliases=["Davis"])
    with session_scope() as db:
        assert [a.id for a in alias_matches(db, owner, "davis")] == [actor]
    # Someone else whose name starts with that word arrives later (allowed: it is not the same name).
    other = cmd(client, "actor.create", name="Davis Smith")
    with session_scope() as db:
        assert alias_matches(db, owner, "Davis") == []
    task = cmd(client, "task.create", title="Which Davis?", assignee="Davis")
    assert task["assignee_id"] not in (actor, other["id"])  # unresolved: a new person, as for any unknown name


def test_actors_cannot_take_an_alias_without_allow_duplicate(client):
    me(client)
    cmd(client, "settings.update", name_aliases=["Davis Dunham"])
    status, err = error(client, "actor.create", name="Davis Dunham")
    assert (status, err["code"]) == (409, "DUPLICATE_ACTOR_NAME")
    sam = cmd(client, "actor.create", name="Sam")
    assert error(client, "actor.update", actor_id=sam["id"], expected_revision=1, name="Davis Dunham")[1]["code"] == \
        "DUPLICATE_ACTOR_NAME"
    twin = cmd(client, "actor.create", name="Davis Dunham", allow_duplicate=True)
    assert error(client, "task.create", title="Which one?", assignee="Davis Dunham")[1]["code"] == "AMBIGUOUS_ACTOR"
    assert cmd(client, "task.create", title="By ID", assignee_id=twin["id"])["assignee_id"] == twin["id"]


def test_aliases_are_personal_workspace_only(client):
    w = client.post("/api/v1/accounts/workspaces", json={"name": "Team", "kind": "space"})
    assert w.status_code == 200, w.text
    switched = client.post("/api/v1/accounts/switch", json={"workspace_id": w.json()["id"]})
    assert switched.status_code == 200, switched.text
    status, err = error(client, "settings.update", name_aliases=["Davis Dunham"])
    assert (status, err["code"]) == (403, "PERSONAL_WORKSPACE")
