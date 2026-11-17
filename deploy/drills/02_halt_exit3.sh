#!/usr/bin/env bash
# Drill 2: a persisted halt makes the next run exit 3 and place no orders.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
seed make "$SB" --halt "drill: operator halt" || fail "drill2: seed"
run_cli --mode dry > "$SB/run.log" 2>&1; rc=$?
[ "$rc" -eq 3 ] || fail "drill2: exit $rc (want 3); see $SB/run.log"
[ "$(seed orders "$SB")" -eq 0 ] || fail "drill2: orders were written while halted"
pass "drill2: halted run exits 3, no orders"
