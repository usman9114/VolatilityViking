#!/usr/bin/env bash
# Drill 3 (VM only): an invalid testnet key must make startup refuse (exit 4), with no orders.
# Run on the VM from the checkout: VV_DRILL_VM=1 bash deploy/drills/03_bad_key_exit4.sh
# Uses dummy key values on purpose; the real .env is not read by this drill.
if [ "${VV_DRILL_VM:-0}" != "1" ]; then
  echo "SKIP drill3: VM only (set VV_DRILL_VM=1 on the VM, testnet mode)"; exit 0
fi
out="$(docker compose run --rm -T -e VV_BINANCE_KEY=drill-invalid -e VV_BINANCE_SECRET=drill-invalid runner --mode testnet 2>&1)"; rc=$?
[ "$rc" -eq 4 ] && echo "PASS drill3: invalid key refused at startup (exit 4)" && exit 0
echo "FAIL drill3: exit $rc (want 4)"; echo "$out" | tail -5; exit 1
