# Shared helpers for the ops drills. Sourced, not run.
# Every drill prints exactly one PASS or FAIL line (or SKIP for VM-only drills run off the VM).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
SB="$(mktemp -d)"
trap 'rm -rf "$SB"' EXIT
# Dry drills run offline: any market-data fetch fails fast, which is the "no network" case.
export HTTPS_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9
seed() { "$PY" "$REPO/deploy/drills/seed.py" "$@"; }
run_cli() { (cd "$REPO" && "$PY" -m vv.runner --config "$SB/runner.yaml" "$@"); }
pass() { echo "PASS $1"; exit 0; }
fail() { echo "FAIL $1"; exit 1; }
