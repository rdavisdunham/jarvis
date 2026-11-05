#!/usr/bin/env bash
# The quick pre-push check for jarvis: what the remotea `quick` verify entry runs after every agent turn
# (docs/AGENT_CONTROL.md § CI as the final verification). GitHub CI on the pull request head is the final
# verification; ./scripts/verify.sh stays for an explicit full local run. Target: under 3 minutes.
#
#   ./scripts/verify-quick.sh [--dry-run] [--no-sync] [--base <ref>]
#
# 1. PATH and the test database as in verify.sh: JARVIS_ENV_FILE="", the caller's JARVIS_DATABASE_URL (pytest makes
#    its own throwaway database there), else a throwaway postgres:16-alpine container removed on exit.
# 2. What changed: files that differ from the merge base with <ref> (default $JARVIS_VERIFY_BASE, else origin/main,
#    else main), committed or not, plus untracked files.
# 3. ruff with CI's selection (E9,F63,F7,F82,F811) on apps/api/jarvis, tests and scripts/app_eval.
# 4. pytest on the tests related to the change (scripts/verify_quick.py: apps/api/jarvis/x.py → tests/test_x*.py,
#    context modules → tests/context/test_y*.py or tests/context, changed test files themselves), always with the
#    drift guards.
# 5. Web, only when apps/web changed: `tsc -b`, then `vitest related --run` for the changed source files.
#
# Stops at the first failure with a non-zero exit. --no-sync (or JARVIS_VERIFY_SYNC=0) skips `uv sync` and
# `npm ci` (a sandboxed entry whose target `prepare` steps installed them). --dry-run prints the plan only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

dry=0
sync="${JARVIS_VERIFY_SYNC:-1}"
base="${JARVIS_VERIFY_BASE:-}"
while (($#)); do
  case "$1" in
    --dry-run) dry=1 ;;
    --no-sync) sync=0 ;;
    --base)
      base="${2:-}"
      shift
      ;;
    --base=*) base="${1#--base=}" ;;
    -h | --help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "verify-quick.sh: unknown argument '$1' (for a full run use ./scripts/verify.sh)" >&2
      exit 2
      ;;
  esac
  shift
done

section() { printf '\n==> %s\n' "$*"; }
run() {
  printf '+ %s\n' "$*"
  if [ "$dry" = 0 ]; then "$@"; fi
}
die() {
  echo "verify-quick.sh: $*" >&2
  exit 1
}
started=$SECONDS

# ── PATH (as verify.sh) ───────────────────────────────────────────────────────
add_path() { if [ -d "$1" ]; then case ":$PATH:" in *":$1:"*) ;; *) PATH="$PATH:$1" ;; esac; fi; }
if [ -n "${HOME:-}" ]; then
  add_path "$HOME/.local/bin"
  add_path "$HOME/.cargo/bin"
  if ! command -v node >/dev/null 2>&1; then
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

# ── What changed ──────────────────────────────────────────────────────────────
section "changes"
export GIT_OPTIONAL_LOCKS=0 # The sandbox mounts the repository's git folder read-only.
command -v git >/dev/null 2>&1 || die "git is not on PATH ($PATH)"
if [ -z "$base" ]; then
  for ref in origin/main main; do
    if git rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
      base="$ref"
      break
    fi
  done
fi
[ -n "$base" ] || die "no base ref (origin/main or main); pass --base <ref>"
merge_base="$(git merge-base HEAD "$base")" || die "no merge base between HEAD and $base"
changed=()
while IFS= read -r path; do
  if [ -n "$path" ] && [ -e "$path" ]; then changed+=("$path"); fi
done < <({
  git diff --name-only "$merge_base"
  git ls-files --others --exclude-standard
} | sort -u)
echo "base $base ($(git rev-parse --short "$merge_base")): ${#changed[@]} changed file(s)"
printf '  %s\n' ${changed[@]+"${changed[@]}"}

py=python3
if [ -x .venv/bin/python ]; then py=.venv/bin/python; fi
web=()
while IFS= read -r path; do
  if [ -n "$path" ]; then web+=("$path"); fi
done < <("$py" scripts/verify_quick.py web ${changed[@]+"${changed[@]}"})
tsc="$("$py" scripts/verify_quick.py tsc ${changed[@]+"${changed[@]}"})"

# ── Test database (as verify.sh) ──────────────────────────────────────────────
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

section "test database"
if [ -n "${JARVIS_DATABASE_URL:-}" ]; then
  echo "using the caller's JARVIS_DATABASE_URL (the session creates its own database there)"
else
  command -v docker >/dev/null 2>&1 || [ "$dry" = 1 ] || die "no JARVIS_DATABASE_URL and no docker to start a throwaway Postgres (inside the remotea sandbox, set JARVIS_DATABASE_URL in the verify entry's env)"
  container="jarvis-verify-quick-$$-${RANDOM}"
  run docker run -d --rm --name "$container" -e POSTGRES_PASSWORD=verify -e POSTGRES_DB=jarvis -p 127.0.0.1::5432 postgres:16-alpine
  if [ "$dry" = 1 ]; then
    port="<port>"
  else
    port="$(docker port "$container" 5432/tcp | head -n 1 | sed 's/.*://')"
    [ -n "$port" ] || die "could not read the throwaway Postgres port"
    for _ in $(seq 1 60); do
      if docker exec "$container" pg_isready -q -h 127.0.0.1 -p 5432 -U postgres -d jarvis; then break; fi
      sleep 1
    done
    docker exec "$container" pg_isready -q -h 127.0.0.1 -p 5432 -U postgres -d jarvis || die "the throwaway Postgres did not become ready"
  fi
  export JARVIS_DATABASE_URL="postgresql+psycopg://postgres:verify@127.0.0.1:${port}/jarvis"
  echo "throwaway Postgres $container on 127.0.0.1:$port (removed on exit)"
fi

# ── API ───────────────────────────────────────────────────────────────────────
if [ "$sync" != 0 ]; then
  section "uv sync --locked --group dev"
  run uv sync --locked --group dev
fi
[ -x .venv/bin/python ] || [ "$dry" = 1 ] || die "no .venv: run without --no-sync, or install it first (uv sync --locked --group dev)"
section "ruff (CI's selection)"
run .venv/bin/python -m ruff check --select E9,F63,F7,F82,F811 apps/api/jarvis tests scripts/app_eval

tests=()
while IFS= read -r path; do
  if [ -n "$path" ]; then tests+=("$path"); fi
done < <("$py" scripts/verify_quick.py api ${changed[@]+"${changed[@]}"})
section "API tests related to the change (${#tests[@]} path(s))"
run .venv/bin/python -m pytest -q --tb=short "${tests[@]}"

# ── Web ───────────────────────────────────────────────────────────────────────
if [ "$tsc" = yes ] || [ ${#web[@]} -gt 0 ]; then
  cd "$ROOT/apps/web"
  if [ ! -d node_modules ]; then
    [ "$sync" != 0 ] || [ "$dry" = 1 ] || die "apps/web has no node_modules: run without --no-sync, or npm ci first"
    section "web: npm ci"
    run npm ci --no-audit --no-fund
  fi
  section "web: type check (tsc -b)"
  run npx --no-install tsc -b
  if [ ${#web[@]} -gt 0 ]; then
    section "web: tests related to the change (${#web[@]} file(s))"
    run npx --no-install vitest related --run --passWithNoTests "${web[@]}"
  fi
else
  section "web: unchanged, skipped"
fi

section "quick check passed in $((SECONDS - started)) s (GitHub CI on the pull request runs the full suite)"
