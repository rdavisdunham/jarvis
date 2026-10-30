#!/usr/bin/env bash
# Repository-local verification for jarvis: what a remotea `verify` entry runs
# (docs/AGENT_CONTROL.md § An execution target with verification).
#
#   ./scripts/verify.sh [--dry-run] [--no-sync] [--only api|web] [pytest args…]
#
# 1. PATH: keeps the caller's PATH first, then adds ~/.local/bin, ~/.cargo/bin,
#    nvm's default node (when no node is on PATH yet) and /usr/local/bin, so
#    `uv`, `node` and `npm` resolve under systemd's minimal service PATH.
# 2. Database: JARVIS_ENV_FILE="" (never the developer's .env) and a disposable
#    test database. JARVIS_DATABASE_URL from the caller is used as given (the
#    test session creates and drops its own `jarvis_test_<uuid>` database on
#    that server, so a shared test server is fine); without it a throwaway
#    postgres:16-alpine container is started on a free loopback port and
#    removed on exit.
# 3. API: `uv sync --locked --group dev`, then the pytest suite (tests/) with the
#    synced venv's python. Extra arguments go to pytest (e.g. -x, -k agent).
# 4. Web (apps/web): `npm ci` when node_modules is missing, `npx tsc -b`,
#    `npx vitest run`.
#
# Stops at the first failure with a non-zero exit. --no-sync (or
# JARVIS_VERIFY_SYNC=0) skips `uv sync` and `npm ci`: for a sandboxed verify
# entry whose dependencies were installed beforehand (remotea target
# `prepare` steps), since the sandbox has no internet. --only (or
# JARVIS_VERIFY_ONLY) runs one half. --dry-run prints the commands only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

dry=0
sync="${JARVIS_VERIFY_SYNC:-1}"
only="${JARVIS_VERIFY_ONLY:-}"
pytest_args=()
while (($#)); do
  case "$1" in
    --dry-run) dry=1 ;;
    --no-sync) sync=0 ;;
    --only)
      only="${2:-}"
      shift
      ;;
    --only=*) only="${1#--only=}" ;;
    -h | --help)
      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) pytest_args+=("$1") ;;
  esac
  shift
done
case "$only" in
  "" | api | web) ;;
  *)
    echo "verify.sh: --only takes api or web, not '$only'" >&2
    exit 2
    ;;
esac

section() { printf '\n==> %s\n' "$*"; }
run() {
  printf '+ %s\n' "$*"
  if [ "$dry" = 0 ]; then "$@"; fi
}
die() {
  echo "verify.sh: $*" >&2
  exit 1
}

# ── PATH ──────────────────────────────────────────────────────────────────────
add_path() { if [ -d "$1" ]; then case ":$PATH:" in *":$1:"*) ;; *) PATH="$PATH:$1" ;; esac; fi; }
if [ -n "${HOME:-}" ]; then
  add_path "$HOME/.local/bin"
  add_path "$HOME/.cargo/bin"
  if ! command -v node >/dev/null 2>&1; then
    # nvm's default alias (e.g. "22": the newest 22.x installed), else the newest node it has.
    nvm="${NVM_DIR:-$HOME/.nvm}"
    want="$(cat "$nvm/alias/default" 2>/dev/null || true)"
    want="${want#v}"
    nvm_node="$(find "$nvm/versions/node" -mindepth 1 -maxdepth 1 -type d -name "v${want}*" 2>/dev/null | sort -V | tail -n 1 || true)"
    if [ -z "$nvm_node" ]; then nvm_node="$(find "$nvm/versions/node" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -n 1 || true)"; fi
    if [ -n "$nvm_node" ]; then add_path "$nvm_node/bin"; fi
  fi
fi
add_path /usr/local/bin
export PATH

section "environment"
need=()
if [ "$only" != web ] && [ "$sync" != 0 ]; then need+=(uv); fi
if [ "$only" != api ]; then need+=(node npm npx); fi
for tool in "${need[@]}"; do
  where="$(command -v "$tool" || true)"
  if [ -z "$where" ]; then
    [ "$dry" = 1 ] || die "$tool is not on PATH ($PATH)"
    where="(not found)"
  fi
  echo "$tool: $where"
done

# ── Test database ─────────────────────────────────────────────────────────────
export JARVIS_ENV_FILE=""
container=""
cleanup() {
  if [ -n "$container" ]; then
    docker rm -f "$container" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$only" != web ]; then
  section "test database"
  if [ -n "${JARVIS_DATABASE_URL:-}" ]; then
    echo "using the caller's JARVIS_DATABASE_URL (the session creates its own database there)"
  else
    command -v docker >/dev/null 2>&1 || [ "$dry" = 1 ] || die "no JARVIS_DATABASE_URL and no docker to start a throwaway Postgres (inside the remotea sandbox, set JARVIS_DATABASE_URL in the verify entry's env)"
    container="jarvis-verify-$$-${RANDOM}"
    run docker run -d --rm --name "$container" -e POSTGRES_PASSWORD=verify -e POSTGRES_DB=jarvis -p 127.0.0.1::5432 postgres:16-alpine
    if [ "$dry" = 1 ]; then
      port="<port>"
    else
      port="$(docker port "$container" 5432/tcp | head -n 1 | sed 's/.*://')"
      [ -n "$port" ] || die "could not read the throwaway Postgres port"
      # Over TCP: the image's init server listens on the socket only, so this waits for the real one.
      for _ in $(seq 1 60); do
        if docker exec "$container" pg_isready -q -h 127.0.0.1 -p 5432 -U postgres -d jarvis; then break; fi
        sleep 1
      done
      docker exec "$container" pg_isready -q -h 127.0.0.1 -p 5432 -U postgres -d jarvis || die "the throwaway Postgres did not become ready"
    fi
    export JARVIS_DATABASE_URL="postgresql+psycopg://postgres:verify@127.0.0.1:${port}/jarvis"
    echo "throwaway Postgres $container on 127.0.0.1:$port (removed on exit)"
  fi
fi

# ── API ───────────────────────────────────────────────────────────────────────
if [ "$only" != web ]; then
  if [ "$sync" != 0 ]; then
    section "uv sync --locked --group dev"
    run uv sync --locked --group dev
  fi
  [ -x .venv/bin/python ] || [ "$dry" = 1 ] || die "no .venv: run without --no-sync, or install it first (uv sync --locked --group dev)"
  section "API tests (pytest)"
  run .venv/bin/python -m pytest -q --tb=short ${pytest_args[@]+"${pytest_args[@]}"}
fi

# ── Web ───────────────────────────────────────────────────────────────────────
if [ "$only" != api ]; then
  cd "$ROOT/apps/web"
  if [ ! -d node_modules ]; then
    [ "$sync" != 0 ] || [ "$dry" = 1 ] || die "apps/web has no node_modules: run without --no-sync, or npm ci first"
    section "web: npm ci"
    run npm ci --no-audit --no-fund
  fi
  section "web: type check (tsc -b)"
  run npx --no-install tsc -b
  section "web: tests (vitest run)"
  run npx --no-install vitest run
fi

section "verify passed"
