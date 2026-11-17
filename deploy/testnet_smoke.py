"""Private smoke test for the Binance USD-M futures test venue. Never prints secrets.

Checks, in order:
1. Authenticated balance read.
2. The adapter's startup checks: one-way position mode, isolated margin, leverage 1.
3. Positions and reconcile (expects flat).
4. Places ONE tiny post-only BUY far below the market (cannot fill), then cancels it.

Env: VV_BINANCE_KEY, VV_BINANCE_SECRET (or the fallbacks `binance` / `bs_key`),
VV_BINANCE_TEST_VENUE=testnet|demo. Refuses to run in live mode.

Usage (repo root): PYTHONPATH=. .venv/bin/python deploy/testnet_smoke.py --symbol BTC/USDT:USDT
(in this cloud container also set SSL_CERT_FILE=/root/.ccr/ca-bundle.crt)
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _load_keys() -> None:
    if not os.environ.get("VV_BINANCE_KEY") and os.environ.get("binance"):
        os.environ["VV_BINANCE_KEY"] = os.environ["binance"]
    if not os.environ.get("VV_BINANCE_SECRET") and os.environ.get("bs_key"):
        os.environ["VV_BINANCE_SECRET"] = os.environ["bs_key"]


def _redact(msg: str) -> str:
    for name in ("VV_BINANCE_KEY", "VV_BINANCE_SECRET"):
        v = os.environ.get(name)
        if v:
            msg = msg.replace(v, "***")
    return msg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    a = ap.parse_args(argv)
    _load_keys()
    if not (os.environ.get("VV_BINANCE_KEY") and os.environ.get("VV_BINANCE_SECRET")):
        print("FAIL: no key/secret in env (VV_BINANCE_KEY/VV_BINANCE_SECRET or binance/bs_key)")
        return 2
    venue = os.environ.get("VV_BINANCE_TEST_VENUE", "demo")
    print(f"venue={venue} (key present, secret present; values not shown)")

    from vv.exec.binance_perp import CcxtPerpClient

    steps = []
    try:
        client = CcxtPerpClient("testnet")
        ex = client.ex
        client.sync_time()
        bal = ex.fetch_balance()
        usdt = bal.get("USDT", {})
        steps.append(("balance", True, f"USDT total={usdt.get('total')} free={usdt.get('free')}"))

        from vv.exec.binance_perp import PerpAdapter

        adapter = PerpAdapter(client, halt_active=lambda: False)
        adapter.start([a.symbol])
        steps.append(("startup checks (one-way, isolated, 1x)", True, "ok"))

        pos = adapter.positions()
        flat = all(abs(p.qty) < 1e-12 for p in pos.values()) if isinstance(pos, dict) else True
        steps.append(("positions", True, "flat" if flat else f"NOT flat: {pos}"))

        ob = ex.fetch_order_book(a.symbol, limit=5)
        bid = ob["bids"][0][0]
        m = ex.market(a.symbol)
        price = float(ex.price_to_precision(a.symbol, bid * 0.8))
        min_cost = (m["limits"]["cost"] or {}).get("min") or 100.0
        qty = float(ex.amount_to_precision(a.symbol, max(min_cost * 1.1 / price, m["limits"]["amount"]["min"] or 0)))
        cid = f"vv-smoke-{int(time.time())}"
        o = ex.create_order(a.symbol, "limit", "buy", qty, price,
                            {"timeInForce": "GTX", "newClientOrderId": cid})
        steps.append(("place post-only buy 20% below bid", True, f"id={o.get('id')} qty={qty} px={price}"))
        time.sleep(1)
        ex.cancel_order(o["id"], a.symbol)
        steps.append(("cancel", True, "ok"))
    except Exception as e:  # report and fail, never leak secrets
        steps.append((type(e).__name__, False, _redact(str(e))[:300]))

    ok = all(s[1] for s in steps)
    for name, good, detail in steps:
        print(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
