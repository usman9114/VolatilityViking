#!/usr/bin/env bash
# Drill 5 (VM only): exchange and data hosts unreachable. The run must alert and place no orders.
# Run on the VM: VV_DRILL_VM=1 bash deploy/drills/05_unreachable_network_alert.sh
if [ "${VV_DRILL_VM:-0}" != "1" ]; then
  echo "SKIP drill5: VM only (set VV_DRILL_VM=1 on the VM)"; exit 0
fi
out="$(docker compose run --rm -T -e HTTPS_PROXY=http://127.0.0.1:9 -e https_proxy=http://127.0.0.1:9 runner --mode testnet 2>&1)"; rc=$?
echo "$out" | grep -q "ALERT" || { echo "FAIL drill5: no alert logged (exit $rc)"; exit 1; }
[ "$rc" -ne 0 ] || { echo "FAIL drill5: run exited 0 with network down"; exit 1; }
echo "PASS drill5: network down -> alert logged, exit $rc (check orders in the runner DB before sign-off)"
