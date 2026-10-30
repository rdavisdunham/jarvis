"""scripts/verify.sh, the jarvis verify command for a remotea target: syntax and --dry-run only (nothing is run)."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify.sh"


def dry_run(**env):
    clean = {k: v for k, v in os.environ.items() if k not in ("JARVIS_DATABASE_URL", "VERIFY_DATABASE_URL")}
    return subprocess.run([str(SCRIPT), "--dry-run"], capture_output=True, text=True, timeout=30, check=False, env={**clean, **env})


def test_the_script_parses_and_is_executable():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, timeout=30, check=False).returncode == 0


def test_dry_run_prints_every_stage_and_never_the_database_password():
    out = dry_run(VERIFY_DATABASE_URL="postgresql+psycopg://jarvis:s3cret@127.0.0.1:5432/jarvis_test")
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    for stage in ("database", "python deps", "pytest", "web deps", "tsc", "vitest"):
        assert any(line.startswith(f"verify: {stage}") for line in lines), stage
    assert "host 127.0.0.1" in out.stdout and "s3cret" not in out.stdout
    assert "python -m pytest tests -q -x" in out.stdout and "npx tsc -b" in out.stdout and "npx vitest run" in out.stdout
    assert "/.local/bin" in out.stdout


def test_dry_run_says_when_no_database_is_set_and_bad_arguments_are_refused():
    assert "unset" in dry_run().stdout
    bad = subprocess.run([str(SCRIPT), "--nope"], capture_output=True, text=True, timeout=30, check=False)
    assert bad.returncode == 2 and "unknown argument" in bad.stderr
