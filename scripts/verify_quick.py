"""Test selection for scripts/verify-quick.sh: which pytest files and web files a change touches.

    python scripts/verify_quick.py api  <changed paths…>   # pytest paths, one per line
    python scripts/verify_quick.py web  <changed paths…>   # web files for `vitest related`, relative to apps/web
    python scripts/verify_quick.py tsc  <changed paths…>   # prints "yes" when the web type check is needed

`select_tests` and `select_web` are pure (tests/test_verify_quick.py). Mapping, for the API:
- a changed test file runs itself (`tests/test_x.py`, `tests/context/test_y.py`);
- `apps/api/jarvis/x.py` runs `tests/test_x*.py`;
- `apps/api/jarvis/context/y.py` runs `tests/context/test_y*.py`, else the whole `tests/context`;
- shared test plumbing (`tests/conftest.py`, `tests/*.py` helpers, fixtures) runs the drift guards;
- anything else (or a module with no matching test) falls back to the drift guards.
The drift guards (`tests/test_drift_guards.py`) always run: they are cheap and catch cross-file mistakes.
"""

import sys
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

DRIFT = "tests/test_drift_guards.py"
API = "apps/api/jarvis/"
CONTEXT = "apps/api/jarvis/context/"
WEB = "apps/web/"
WEB_SOURCES = (".ts", ".tsx", ".css", ".js", ".jsx")
WEB_CONFIG = ("package.json", "package-lock.json", "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json",
              "vite.config.ts", "vitest.config.ts")


def select_tests(changed, tests):
    """Pytest paths for the changed files. `tests`: every test file in the repository (posix, relative to the root)."""
    tests = sorted(set(tests))
    picked = {DRIFT}
    for path in changed:
        name = PurePosixPath(path).name
        if path.startswith("tests/") and name.startswith("test_") and path.endswith(".py"):
            if path in tests:
                picked.add(path)
        elif path.startswith(CONTEXT) and path.endswith(".py"):
            stem = PurePosixPath(path).stem
            matches = [t for t in tests if fnmatch(t, f"tests/context/test_{stem}*.py")]
            picked.update(matches or (["tests/context"] if stem != "__init__" else []))
        elif path.startswith(API) and path.endswith(".py"):
            stem = PurePosixPath(path).stem
            picked.update(t for t in tests if fnmatch(t, f"tests/test_{stem}*.py") and "/" not in t[len("tests/"):])
    return sorted(picked, key=lambda p: (p != DRIFT, p))


def web_changes(changed):
    return [p for p in changed if p.startswith(WEB) and "/node_modules/" not in p]


def select_web(changed):
    """Changed web source files, relative to apps/web, for `vitest related` (it finds the tests that import them)."""
    return sorted(p[len(WEB):] for p in web_changes(changed) if p.endswith(WEB_SOURCES) and p.startswith(WEB + "src/"))


def needs_tsc(changed):
    return any(p.endswith(WEB_SOURCES) or PurePosixPath(p).name in WEB_CONFIG for p in web_changes(changed))


def main(argv):
    if len(argv) < 2 or argv[1] not in ("api", "web", "tsc"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    changed = [a.strip() for a in argv[2:] if a.strip()]
    if argv[1] == "api":
        root = Path(__file__).resolve().parent.parent
        tests = [p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")]
        print("\n".join(select_tests(changed, tests)))
    elif argv[1] == "web":
        print("\n".join(select_web(changed)))
    else:
        print("yes" if needs_tsc(changed) else "no")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
