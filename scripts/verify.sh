#!/usr/bin/env bash
# The jarvis verify command for a remotea execution target (docs/AGENT_CONTROL.md, "An execution target with
# verification"). remotea runs it as fixed argv in the run's worktree, with no shell profile, the service's minimal
# PATH and, where bubblewrap works, inside a sandbox whose HOME is an empty /tmp/home. So this script sets up what
# it needs itself and never reads the developer's .env or the real database.
#
#   scripts/verify.sh            run every stage, stop at the first failure (exit 1; 2 for a setup refusal)
#   scripts/verify.sh --dry-run  print the stages it would run, run nothing
#
# Environment:
#   JARVIS_DATABASE_URL  the Postgres the test suite connects to (it creates and drops its own jarvis_test_* database
#                        there, so the role needs CREATEDB); defaults to VERIFY_DATABASE_URL
#   VERIFY_DATABASE_URL  the local test Postgres to use when JARVIS_DATABASE_URL is unset
#   VERIFY_ALLOW_REMOTE_DB=1  allow a database host other than localhost / 127.0.0.1 / ::1 / a unix socket
set -uo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "verify: unknown argument $arg (try --dry-run)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

# The sandbox's HOME is empty; the tools live under the service user's real home (mounted read-only there).
REAL_HOME="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
for home in "${REAL_HOME:-}" "${HOME:-}"; do
  [ -n "$home" ] || continue
  [ "$home" = "${REAL_HOME:-}" ] && [ "$home" = "${HOME:-}" ] && [ "${added:-}" = 1 ] && continue
  added=1
  PATH="$home/.local/bin:$PATH"
  # uv-managed interpreters (uv python install): ~/.local/share/uv/python/<version>/bin
  for dir in "$home"/.local/share/uv/python/*/bin; do
    [ -d "$dir" ] && PATH="$dir:$PATH"
  done
done
export PATH
export JARVIS_ENV_FILE=""  # Never load a developer .env (it may name the real database).
export CI="${CI:-1}"

if [ -z "${JARVIS_DATABASE_URL:-}" ]; then
  JARVIS_DATABASE_URL="${VERIFY_DATABASE_URL:-}"
fi
export JARVIS_DATABASE_URL

# The database host part of a SQLAlchemy URL (scheme://user:pass@host:port/db?...), without user, port or path.
db_host() {
  local rest="${1#*://}"
  rest="${rest%%/*}"
  rest="${rest##*@}"
  case "$rest" in
    \[*\]*) rest="${rest#[}"; rest="${rest%%]*}" ;;
    *) rest="${rest%%:*}" ;;
  esac
  printf '%s' "$rest"
}

STAGES=("database" "python deps" "pytest" "web deps" "tsc" "vitest")
if [ "$DRY_RUN" = 1 ]; then
  echo "verify: dry run in $ROOT (nothing is run)"
  if [ -n "$JARVIS_DATABASE_URL" ]; then  # Never print the URL itself: it carries the password.
    echo "verify: database   JARVIS_DATABASE_URL set, host $(db_host "$JARVIS_DATABASE_URL" | sed 's/^$/(unix socket)/'); JARVIS_ENV_FILE empty"
  else
    echo "verify: database   JARVIS_DATABASE_URL unset: set it or VERIFY_DATABASE_URL (the run would stop here)"
  fi
  echo "verify: python deps $([ -d .venv ] && echo ".venv present, no install" || echo "uv sync --locked --group dev")"
  echo "verify: pytest     .venv/bin/python -m pytest tests -q -x"
  echo "verify: web deps   apps/web: $([ -d apps/web/node_modules ] && echo "node_modules present, no install" || echo "npm ci --no-audit --no-fund")"
  echo "verify: tsc        apps/web: npx tsc -b"
  echo "verify: vitest     apps/web: npx vitest run"
  echo "verify: PATH starts $(printf '%s' "$PATH" | cut -d: -f1-4)"
  exit 0
fi

started=$SECONDS
stage() {  # stage <name> <command...>: run it, print one summary line, stop the script on failure.
  local name="$1"; shift
  local at=$SECONDS
  "$@"
  local code=$?
  if [ "$code" -ne 0 ]; then
    echo "verify: FAIL $name (exit $code, $((SECONDS - at))s)"
    echo "verify: stopped at $name; later stages did not run"
    exit 1
  fi
  echo "verify: ok   $name ($((SECONDS - at))s)"
}

if [ -z "$JARVIS_DATABASE_URL" ]; then
  echo "verify: FAIL database (no JARVIS_DATABASE_URL or VERIFY_DATABASE_URL: point it at a disposable local test Postgres)"
  exit 2
fi
host="$(db_host "$JARVIS_DATABASE_URL")"
case "$host" in
  ""|localhost|127.0.0.1|::1) ;;
  *)
    if [ "${VERIFY_ALLOW_REMOTE_DB:-}" != 1 ]; then
      echo "verify: FAIL database (host $host is not local; refusing so the suite never touches a real database)"
      exit 2
    fi ;;
esac
echo "verify: ok   database (host ${host:-unix socket}; JARVIS_ENV_FILE empty)"

if [ -d .venv ]; then
  echo "verify: ok   python deps (.venv present)"
else
  stage "python deps" uv sync --locked --group dev
fi
stage "pytest" .venv/bin/python -m pytest tests -q -x

cd apps/web || exit 2
if [ -d node_modules ]; then
  echo "verify: ok   web deps (node_modules present)"
else
  stage "web deps" npm ci --no-audit --no-fund
fi
stage "tsc" npx tsc -b
stage "vitest" npx vitest run

echo "verify: passed ${#STAGES[@]} stages in $((SECONDS - started))s"
