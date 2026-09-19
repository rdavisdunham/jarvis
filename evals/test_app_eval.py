"""Meta-evals: fixtures, guardrails and graders must be trustworthy."""

import asyncio
import copy
import json
import os
from uuid import uuid4

import pytest

from scripts.app_eval import catalog
from scripts.app_eval.environment import DEFAULT_URL, settings_env, validate_url
from scripts.app_eval.model_runner import PROBES, grade


def test_every_feature_has_25_to_50_distinct_evidence_based_cases():
    result = catalog.validate()
    assert result["features"] == 40 and result["cases"] == 1001


def test_all_executable_case_ids_exist_in_catalog():
    from scripts.app_eval.contracts import CASES

    _, cases = catalog.load()
    identities = {c["id"] for c in cases}
    assert set(CASES) <= identities and set(PROBES) <= identities


def test_surface_inventory_matches_current_app():
    from scripts.app_eval.inventory import check

    check()


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://eridani_eval:pw@production.example/eridani_eval_corpus",
        "postgresql+psycopg://jarvis:pw@localhost/eridani_eval_corpus",
        "postgresql+psycopg://eridani_eval:pw@localhost/jarvis",
        "postgresql+psycopg://eridani_eval:pw@localhost/postgres",
        "postgresql+psycopg://eridani_eval:pw@localhost/eridani_eval_corpus?host=production.example",
        "sqlite://",
        "postgresql+psycopg://eridani_eval:pw@localhost/eridani_eval_trial_not-a-uuid",
    ],
)
def test_remote_production_and_namespace_bypasses_are_refused(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_corpus_cannot_be_used_as_a_disposable_trial():
    with pytest.raises(ValueError):
        validate_url(DEFAULT_URL, trial=True)


def test_default_environment_contains_no_real_provider_credentials():
    values = settings_env(DEFAULT_URL)
    assert values["JARVIS_ENV_FILE"] == ""
    assert values["JARVIS_EXTERNAL_SERVICES_ENABLED"] == "false"
    assert not any(
        values[k]
        for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "JARVIS_OPENAI_API_KEY", "JARVIS_GEMINI_API_KEY")
    )
    assert values["JARVIS_WORKER_ENABLED"] == "false"


def state():
    return {
        "tasks": [{"id": "existing", "owner_id": "eval-rowan", "title": "Keep", "notes": "", "revision": 1}],
        "notes": [],
        "memories": [],
        "schedules": [],
        "owner_settings": [],
    }


def test_positive_and_negative_create_oracles():
    before = state()
    after = copy.deepcopy(before)
    after["tasks"].append(
        {
            "id": "new",
            "owner_id": "eval-rowan",
            "title": "Buy oat milk",
            "status": "open",
            "due_date": None,
            "due_time": None,
            "completed_at": None,
        }
    )
    probe = PROBES["task_capture.01"]
    assert grade(probe, before, after, {}, "succeeded", {}) == []
    assert grade(probe, before, before, {}, "succeeded", {"message": "Done!"})
    after["tasks"].append(dict(after["tasks"][-1], id="duplicate"))
    assert grade(probe, before, after, {}, "succeeded", {})


def test_grader_rejects_unrequested_mutations_and_private_leaks():
    before = state()
    after = copy.deepcopy(before)
    after["tasks"][0]["title"] = "Wrong"
    after["memories"] = [{"id": "leak", "owner_id": "eval-jules", "content": "Changed"}]
    failures = grade({"needs_input": True}, before, after, {}, "needs_input", "JULES-PRIVATE-LANTERN-7392")
    assert any("Unrequested task" in f for f in failures)
    assert any("private account" in f for f in failures)
    assert any("canary" in f for f in failures)


def test_clarification_requires_real_queue_state_and_preserves_data():
    before = state()
    assert grade({"needs_input": True}, before, before, {}, "needs_input", "Which proposal?") == []
    assert grade({"needs_input": True}, before, before, {}, "succeeded", "Which proposal?")
    assert grade({"needs_input": True}, before, before, {}, "needs_input", "")


def test_contract_pass_does_not_claim_acceptance_pass():
    report = catalog.report([{"case_id": "task_capture.01", "layer": "command_contract", "status": "passed"}])
    assert report["summary"] == {"not_run": 1001}


def test_oracle_persona_is_not_a_model_prompt():
    import inspect

    from scripts.app_eval.model_runner import execute_probe

    source = inspect.getsource(execute_probe)
    assert "oracle_only" not in source and "persona" not in source
    assert all("oracle_only" not in json.dumps(p) for p in PROBES.values())


def test_probe_targets_exist_and_named_existing_records_match():
    corpus = json.loads((catalog.CATALOG / "corpus/rowan-v1.json").read_text())
    refs = {t["key"]: t for t in corpus["tasks"]}
    for probe in PROBES.values():
        if probe.get("target"):
            assert probe["target"] in refs
            assert refs[probe["target"]]["title"].casefold() in probe["message"].casefold()


@pytest.mark.skipif(
    os.environ.get("ERIDANI_EVAL_INTEGRATION") != "1", reason="Explicit local synthetic PG integration only"
)
def test_real_queue_adapter_and_trial_cleanup():
    from jarvis.config import get_settings
    from sqlalchemy import text

    from scripts.app_eval.environment import admin_for, environment, trial_database
    from scripts.app_eval.model_runner import execute_probe

    async def response(agent, messages, definitions, **kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "id": str(uuid4()),
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": str(uuid4()),
                                    "type": "function",
                                    "function": {
                                        "name": "task_create",
                                        "arguments": json.dumps({"title": "Buy oat milk"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        return {
            "id": str(uuid4()),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"message": {"role": "assistant", "content": "Added Buy oat milk."}}],
        }

    with trial_database(DEFAULT_URL) as (url, fixture), environment(url):
        name = validate_url(url, trial=True).database
        os.environ["JARVIS_OPENAI_API_KEY"] = "synthetic-only"
        os.environ["JARVIS_EXTERNAL_SERVICES_ENABLED"] = "true"
        get_settings.cache_clear()
        trace = []
        outcome = asyncio.run(
            execute_probe(
                "task_capture.01", PROBES["task_capture.01"], "luna", fixture, trace, scripted=response
            )
        )
        assert outcome["status"] == "passed", outcome["failures"]
        assert any(t["name"] == "task_create" for t in trace)
    admin = admin_for(DEFAULT_URL)
    try:
        with admin.connect() as db:
            assert not db.scalar(text("select 1 from pg_database where datname=:name"), {"name": name})
    finally:
        admin.dispose()


def test_synthetic_encryption_key_is_stable_across_clones():
    assert (
        settings_env(DEFAULT_URL)["JARVIS_INTEGRATION_ENCRYPTION_KEY"]
        == settings_env(DEFAULT_URL)["JARVIS_INTEGRATION_ENCRYPTION_KEY"]
    )


def test_offline_plugin_blocks_dns_and_raw_external_ip():
    import socket

    from evals import offline

    offline.pytest_configure(None)
    try:
        with pytest.raises(RuntimeError, match="external DNS"):
            socket.getaddrinfo("example.com", 443)
        with socket.socket() as sock, pytest.raises(RuntimeError, match="external socket"):
            sock.connect(("192.0.2.1", 443))
        assert socket.getaddrinfo("127.0.0.1", 54340)
    finally:
        offline.pytest_unconfigure(None)


@pytest.mark.skipif(
    os.environ.get("ERIDANI_EVAL_INTEGRATION") != "1", reason="Explicit local synthetic PG integration only"
)
def test_seeded_receipts_are_decryptable_in_a_fresh_clone():
    from jarvis.db import session_scope
    from jarvis.models import ActionChange
    from jarvis.work_crypto import unseal
    from sqlalchemy import select

    from scripts.app_eval.environment import environment, trial_database

    with trial_database(DEFAULT_URL) as (url, _fixture), environment(url), session_scope() as db:
        change = db.scalar(select(ActionChange).where(ActionChange.after_ciphertext.is_not(None)))
        assert change is not None
        assert unseal(change.after_ciphertext)
