#!/usr/bin/env bash
# Drill 4: yesterday's daily bar is missing and cannot be fetched. Exit 2, no orders.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
seed make "$SB" --drop-yesterday || fail "drill4: seed"
run_cli --mode dry > "$SB/run.log" 2>&1; rc=$?
[ "$rc" -eq 2 ] || fail "drill4: exit $rc (want 2); see $SB/run.log"
[ "$(seed orders "$SB")" -eq 0 ] || fail "drill4: orders were written on a data gap"
pass "drill4: missing bar exits 2, no orders"
