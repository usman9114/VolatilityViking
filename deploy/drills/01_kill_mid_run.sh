#!/usr/bin/env bash
# Drill 1: kill the runner mid-run; the next run must start cleanly and finish with no halt.
# Dry mode here (kill timing is best effort: the run is fast, so the kill may land after it ends).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
seed make "$SB" || fail "drill1: seed"
run_cli --mode dry > "$SB/killed.log" 2>&1 &
pid=$!
for _ in $(seq 1 2000); do
  [ "$(seed runs "$SB")" -ge 1 ] && break
  kill -0 "$pid" 2>/dev/null || break
done
kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
run_cli --mode dry > "$SB/next.log" 2>&1; rc=$?
[ "$rc" -eq 0 ] || fail "drill1: next run exit $rc (want 0); see $SB/next.log"
[ "$(seed halted "$SB")" -eq 0 ] || fail "drill1: halt set after killed run"
[ "$(seed runs "$SB")" -ge 2 ] || fail "drill1: next run not recorded"
pass "drill1: next run after kill exits 0, no halt"
