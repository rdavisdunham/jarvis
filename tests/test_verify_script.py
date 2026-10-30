"""scripts/verify.sh, the jarvis verify command for a remotea target: syntax and --dry-run only (nothing is run)."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify.sh"


def dry_run(*args, **env):
    clean = {k: v for k, v in os.environ.items() if k not in ("JARVIS_DATABASE_URL", "JARVIS_VERIFY_SYNC", "JARVIS_VERIFY_ONLY")}
    return subprocess.run([str(SCRIPT), "--dry-run", *args], capture_output=True, text=True, timeout=30, check=False,
                          env={**clean, **env})


def test_the_script_parses_and_is_executable():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, timeout=30, check=False).returncode == 0


def test_dry_run_prints_every_section_and_never_the_database_url():
    out = dry_run(JARVIS_DATABASE_URL="postgresql+psycopg://jarvis:s3cret@127.0.0.1:5432/jarvis_test")
    assert out.returncode == 0, out.stderr
    for section in ("==> environment", "==> test database", "==> API tests (pytest)", "==> verify passed"):
        assert section in out.stdout, section
    assert "-m pytest" in out.stdout and "tsc -b" in out.stdout and "vitest run" in out.stdout
    assert "s3cret" not in out.stdout and "docker run" not in out.stdout


def test_only_and_no_sync_narrow_the_plan_and_a_bad_only_is_refused():
    web = dry_run("--only", "web", "--no-sync")
    assert web.returncode == 0, web.stderr
    assert "-m pytest" not in web.stdout and "uv sync" not in web.stdout
    assert "vitest run" in web.stdout
    bad = dry_run("--only", "nope")
    assert bad.returncode == 2 and "--only takes api or web" in bad.stderr
