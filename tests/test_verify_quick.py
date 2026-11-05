"""scripts/verify_quick.py: which tests the quick pre-push check (scripts/verify-quick.sh) runs for a change."""

import os
import subprocess
from pathlib import Path

from scripts.verify_quick import DRIFT, main, needs_tsc, select_tests, select_web

TESTS = ["tests/test_drift_guards.py", "tests/test_agent_judge.py", "tests/test_run_ci.py", "tests/test_run_coordinator.py",
         "tests/test_run_coordinator_sweep.py", "tests/test_run.py", "tests/context/test_engine.py",
         "tests/context/test_engine_replay.py", "tests/context/test_apply.py"]


def test_a_module_runs_its_own_test_files():
    assert select_tests(["apps/api/jarvis/run_coordinator.py"], TESTS) == [
        DRIFT, "tests/test_run_coordinator.py", "tests/test_run_coordinator_sweep.py"]
    assert select_tests(["apps/api/jarvis/run_ci.py"], TESTS) == [DRIFT, "tests/test_run_ci.py"]


def test_a_context_module_runs_its_tests_or_the_context_suite():
    assert select_tests(["apps/api/jarvis/context/engine.py"], TESTS) == [
        DRIFT, "tests/context/test_engine.py", "tests/context/test_engine_replay.py"]
    assert select_tests(["apps/api/jarvis/context/pipeline.py"], TESTS) == [DRIFT, "tests/context"]
    assert select_tests(["apps/api/jarvis/context/__init__.py"], TESTS) == [DRIFT]
    assert "tests/test_engine.py" not in select_tests(["apps/api/jarvis/engine.py"], TESTS)


def test_changed_tests_run_themselves_and_everything_else_falls_back_to_the_drift_guards():
    assert select_tests(["tests/test_agent_judge.py", "tests/context/test_apply.py"], TESTS) == [
        DRIFT, "tests/context/test_apply.py", "tests/test_agent_judge.py"]
    assert select_tests(["tests/test_deleted.py"], TESTS) == [DRIFT]
    for other in (["README.md"], ["tests/conftest.py"], ["migrations/versions/0051_x.py"], ["apps/api/jarvis/nothing.py"], []):
        assert select_tests(other, TESTS) == [DRIFT]


def test_web_files_go_to_vitest_related_and_the_type_check():
    changed = ["apps/web/src/RunCiCard.tsx", "apps/web/src/runs.css", "apps/web/src/run-ci.test.tsx",
               "apps/web/package.json", "apps/api/jarvis/run_ci.py", "apps/web/node_modules/x/index.js"]
    assert select_web(changed) == ["src/RunCiCard.tsx", "src/run-ci.test.tsx", "src/runs.css"]
    assert needs_tsc(changed) and needs_tsc(["apps/web/package.json"])
    assert not needs_tsc(["apps/api/jarvis/run_ci.py", "README.md"]) and select_web(["README.md"]) == []


def test_the_cli_lists_real_test_files(capsys):
    assert main(["verify_quick.py", "api", "apps/api/jarvis/run_ci.py"]) == 0
    out = capsys.readouterr().out.split()
    assert out[0] == DRIFT and "tests/test_run_ci.py" in out and all(Path(p).exists() for p in out)
    assert main(["verify_quick.py", "tsc", "apps/web/src/App.tsx"]) == 0 and capsys.readouterr().out.strip() == "yes"
    assert main(["verify_quick.py"]) == 2


# ---------------------------------------------------------------- scripts/verify-quick.sh (syntax and --dry-run only)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify-quick.sh"


def test_the_quick_script_parses_and_is_executable():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, timeout=30, check=False).returncode == 0


def test_quick_dry_run_plans_ruff_related_tests_and_never_prints_the_database_url():
    clean = {k: v for k, v in os.environ.items() if k not in ("JARVIS_VERIFY_SYNC", "JARVIS_VERIFY_BASE")}
    out = subprocess.run([str(SCRIPT), "--dry-run", "--no-sync", "--base", "HEAD"], capture_output=True, text=True,
                         timeout=60, check=False,
                         env={**clean, "JARVIS_DATABASE_URL": "postgresql+psycopg://jarvis:s3cret@127.0.0.1:5432/x"})
    assert out.returncode == 0, out.stderr
    for section in ("==> changes", "==> test database", "==> ruff (CI's selection)", "==> API tests related to the change",
                    "quick check passed"):
        assert section in out.stdout, section
    assert "--select E9,F63,F7,F82,F811" in out.stdout and DRIFT in out.stdout
    assert "s3cret" not in out.stdout and "uv sync" not in out.stdout and "docker run" not in out.stdout
    bad = subprocess.run([str(SCRIPT), "-x"], capture_output=True, text=True, timeout=30, check=False)
    assert bad.returncode == 2 and "./scripts/verify.sh" in bad.stderr
