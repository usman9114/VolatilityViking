import logging
import random

import pytest

from vv.core.types import SymbolFilters
from vv.exec.binance_perp import PerpAdapter
from vv.exec.binance_spot import SpotAdapter
from vv.exec.common import (
    DuplicateOrder,
    SecretRedactingFilter,
    StartupCheckFailed,
    TradingHalted,
    client_order_id,
    clip_targets,
)
from vv.exec.fake import FakeVenue
from vv.exec.reconcile import reconcile

F = SymbolFilters(tick_size=0.1, step_size=0.001, min_qty=0.001, min_notional=5.0)
SYMS = ["BTC", "ETH", "SOL"]


def venue(mode="perp", wallet=10_000.0, **kw):
    v = FakeVenue(mode=mode, wallet=wallet, filters={s: F for s in SYMS},
                  quotes={"BTC": (99.9, 100.1), "ETH": (49.9, 50.1), "SOL": (9.9, 10.1)}, **kw)
    return v


def perp(v=None, halt=lambda: False, **kw):
    v = v or venue()
    a = PerpAdapter(v, halt_active=halt, sleep=lambda s: None, **kw)
    a.start(SYMS)
    return v, a


def test_clip_caps_hold_random():
    rng = random.Random(1)
    for _ in range(500):
        prices = {s: rng.uniform(0.5, 500) for s in SYMS}
        t = {s: rng.uniform(-300, 300) for s in SYMS}
        eq = rng.uniform(100, 5000)
        gc, sf = rng.uniform(0.2, 1.5), rng.uniform(0.0, 1.0)
        c = clip_targets(t, prices, eq, gross_cap=gc, max_short_frac=sf)
        gross = sum(abs(q) * prices[s] for s, q in c.items())
        short = sum(-q * prices[s] for s, q in c.items() if q < 0)
        assert gross <= gc * eq * (1 + 1e-9)
        assert short <= sf * eq * (1 + 1e-9)
        for s in SYMS:
            assert abs(c[s]) <= abs(t[s]) + 1e-12 and c[s] * t[s] >= 0


def test_rebalance_respects_caps_end_to_end():
    v, a = perp()
    a.rebalance({"BTC": 500.0, "ETH": -500.0}, decision_ts=1)
    pos = v.positions()
    gross = sum(abs(p.qty) * p.mark for p in pos.values())
    short = sum(-p.qty * p.mark for p in pos.values() if p.qty < 0)
    assert gross <= 10_000 * 1.0 + 1
    assert short <= 5_000 + 1


def test_rounding_never_increases_exposure():
    rng = random.Random(2)
    for i in range(200):
        v, a = perp()
        cur = rng.uniform(-20, 20)
        v.qty["BTC"], v.entry["BTC"] = round(cur, 3), 100.0
        tgt = rng.uniform(-30, 30)
        a.rebalance({"BTC": tgt}, decision_ts=i)
        final = v.qty.get("BTC", 0.0)
        c = clip_targets({"BTC": tgt}, {"BTC": 100.0}, v.balance().margin + 0, gross_cap=1, max_short_frac=0.5)
        # final lies between current and (clipped, unrounded) target: never past the target
        lo, hi = sorted((round(cur, 3), c["BTC"]))
        assert lo - 1e-6 <= final <= hi + 1e-6 or abs(final) <= abs(cur) + 1e-6


def test_below_min_notional_skipped():
    v, a = perp()
    res = a.rebalance({"SOL": 0.2}, decision_ts=1)  # 0.2 * 10 = 2 < 5
    assert res == [] and v.placed == []


def test_exits_and_reductions_reduce_only_entries_not():
    v, a = perp()
    a.rebalance({"BTC": 10.0}, decision_ts=1)
    assert all(not o.reduce_only for o in v.placed)
    n = len(v.placed)
    a.rebalance({"BTC": 4.0}, decision_ts=2)
    assert all(o.reduce_only for o in v.placed[n:])
    n = len(v.placed)
    a.rebalance({"BTC": 0.0}, decision_ts=3)
    assert v.qty["BTC"] == 0 and all(o.reduce_only for o in v.placed[n:])


def test_flip_splits_reduce_then_entry():
    v, a = perp()
    a.rebalance({"BTC": 10.0}, decision_ts=1)
    n = len(v.placed)
    a.rebalance({"BTC": -5.0}, decision_ts=2)
    legs = v.placed[n:]
    assert legs[0].reduce_only and legs[0].qty == 10.0
    assert not legs[1].reduce_only and legs[1].qty == 5.0
    assert v.qty["BTC"] == -5.0


def test_ioc_fallback_when_maker_unfilled():
    v, a = perp(venue(postonly_fills=False))
    res = a.rebalance({"BTC": 10.0}, decision_ts=1)
    assert [r.stage for r in res] == ["maker", "ioc"]
    assert res[0].status == "canceled" and res[1].status == "filled"
    assert res[1].price <= 100.1 * 1.002 + 0.1
    assert v.qty["BTC"] == 10.0


def test_idempotent_client_ids_and_duplicate_rejected():
    assert client_order_id("BTC", "buy", 1.5, 1000) == client_order_id("BTC", "buy", 1.5, 1000)
    assert client_order_id("BTC", "buy", 1.5, 1000) != client_order_id("BTC", "buy", 1.5, 1001)
    assert client_order_id("BTC", "buy", 1.5, 1000).startswith("vv-")
    v, a = perp()
    cid = client_order_id("BTC", "buy", 1.0, 5)
    v.place("BTC", "buy", 1.0, 99.9, tif="GTX", reduce_only=False, client_id=cid)
    with pytest.raises(DuplicateOrder):
        v.place("BTC", "buy", 1.0, 99.9, tif="GTX", reduce_only=False, client_id=cid)


def test_retry_same_decision_does_not_double_trade():
    v, a = perp()
    a.rebalance({"BTC": 10.0}, decision_ts=7)
    a.rebalance({"BTC": 10.0}, decision_ts=7)
    assert v.qty["BTC"] == 10.0 and len(v.placed) == 1


def test_adapter_treats_duplicate_as_already_submitted():
    v, a = perp(venue(postonly_fills=False))
    a.rebalance({"BTC": 10.0}, decision_ts=9, dry_run=False)
    q = v.qty["BTC"]
    v.positions = lambda: {}  # simulate a stale read after a crash: retry re-sends same ids
    a.rebalance({"BTC": 10.0}, decision_ts=9)
    assert v.qty["BTC"] == q


def test_halt_blocks_every_order():
    flag = {"h": True}
    v, a = perp(halt=lambda: flag["h"])
    with pytest.raises(TradingHalted):
        a.rebalance({"BTC": 10.0}, decision_ts=1)
    assert v.placed == []
    flag["h"] = False
    a.rebalance({"BTC": 10.0}, decision_ts=1)
    assert v.placed


def test_halt_mid_run_stops_further_orders():
    calls = {"n": 0}

    def halt():
        calls["n"] += 1
        return calls["n"] > 3  # passes guard + first legs, then trips

    v, a = perp(halt=halt)
    with pytest.raises(TradingHalted):
        a.rebalance({"BTC": 5.0, "ETH": 10.0, "SOL": 20.0}, decision_ts=1)
    assert len(v.placed) < 3


def test_dry_run_places_nothing():
    v, a = perp()
    res = a.rebalance({"BTC": 5.0}, decision_ts=1, dry_run=True)
    assert res and res[0].status == "dry_run" and v.placed == []


def test_mode_guards_refuse():
    v = venue(hedge_mode=True)
    v.qty["BTC"], v.entry["BTC"] = 1.0, 100.0
    a = PerpAdapter(v, halt_active=lambda: False)
    with pytest.raises(StartupCheckFailed):
        a.start(SYMS)
    with pytest.raises(StartupCheckFailed):
        a.rebalance({"BTC": 1.0}, decision_ts=1)

    v = venue()
    v.qty["ETH"], v.entry["ETH"] = 1.0, 50.0  # cross-margin position open
    a = PerpAdapter(v, halt_active=lambda: False)
    with pytest.raises(StartupCheckFailed):
        a.start(SYMS)

    a = PerpAdapter(venue(), halt_active=lambda: False)  # never started
    with pytest.raises(StartupCheckFailed):
        a.rebalance({"BTC": 1.0}, decision_ts=1)


def test_startup_sets_modes_when_safe():
    v = venue(hedge_mode=True)
    a = PerpAdapter(v, halt_active=lambda: False)
    a.start(SYMS)
    assert not v.hedge_mode and v.time_synced
    assert all(v.margin_mode[s] == "isolated" and v.leverage[s] == 1 for s in SYMS)


def test_spot_long_only():
    v = venue("spot")
    a = SpotAdapter(v, halt_active=lambda: False, sleep=lambda s: None)
    a.start()
    a.rebalance({"BTC": -5.0, "ETH": 20.0}, decision_ts=1)
    assert v.qty.get("BTC", 0) == 0 and v.qty["ETH"] > 0
    assert all(not o.reduce_only for o in v.placed)
    a.rebalance({"ETH": 0.0}, decision_ts=2)
    assert v.qty["ETH"] == 0 and v.wallet > 9_000


def test_fake_liq_price_and_funding():
    v = venue()
    v.qty["BTC"], v.entry["BTC"], v.leverage["BTC"] = 1.0, 100.0, 1
    assert v.liq_price("BTC") == pytest.approx(100 * 0.004)
    v.qty["BTC"] = -1.0
    assert v.liq_price("BTC") == pytest.approx(100 * 1.996)
    w = v.wallet
    v.apply_funding(0.01)  # shorts receive
    assert v.wallet == pytest.approx(w + 1.0)


def test_reconcile():
    assert reconcile({"BTC": 1.0}, {"BTC": 1.0}) == []
    d = reconcile({"BTC": 1.0, "ETH": 2.0}, {"BTC": 0.0, "SOL": 3.0})
    kinds = {x.symbol: x.kind for x in d}
    assert kinds == {"BTC": "unexpected", "ETH": "unexpected", "SOL": "missing"}
    assert [x.kind for x in reconcile({"BTC": 1.0}, {"BTC": 2.0})] == ["size"]
    assert reconcile({"BTC": 1.0}, {"BTC": 0.0}, inflight={"BTC": 1.0}) == []


def test_secrets_never_logged(caplog, monkeypatch):
    monkeypatch.setenv("VV_BINANCE_KEY", "KEY-abcdef123456")
    monkeypatch.setenv("VV_BINANCE_SECRET", "SECRET-zyxwvu987654")
    log = logging.getLogger("vv.exec")
    assert any(isinstance(f, SecretRedactingFilter) for f in log.filters)
    with caplog.at_level(logging.INFO, logger="vv.exec"):
        log.error("auth failed key=%s secret=%s", "KEY-abcdef123456", "SECRET-zyxwvu987654")
        log.error("GET /fapi?x=1&signature=" + "a" * 64)
    text = caplog.text
    assert "KEY-abcdef123456" not in text and "SECRET-zyxwvu987654" not in text
    assert "a" * 64 not in text and "***" in text


def test_client_repr_has_no_keys(monkeypatch):
    from vv.exec.binance_perp import CcxtPerpClient

    monkeypatch.setenv("VV_BINANCE_KEY", "KEY-abcdef123456")
    monkeypatch.setenv("VV_BINANCE_SECRET", "SECRET-zyxwvu987654")
    c = CcxtPerpClient("testnet")
    assert "KEY-" not in repr(c) and "SECRET-" not in repr(c)


def test_test_venue_selects_endpoint(monkeypatch):
    import pytest as _pytest

    from vv.exec.binance_perp import CcxtPerpClient

    monkeypatch.delenv("VV_BINANCE_KEY", raising=False)
    monkeypatch.delenv("VV_BINANCE_SECRET", raising=False)
    monkeypatch.setenv("VV_BINANCE_TEST_VENUE", "demo")
    assert "demo-fapi.binance.com" in CcxtPerpClient("testnet").ex.urls["api"]["fapiPrivate"]
    monkeypatch.setenv("VV_BINANCE_TEST_VENUE", "testnet")
    assert "testnet.binancefuture.com" in CcxtPerpClient("testnet").ex.urls["api"]["fapiPrivate"]
    monkeypatch.setenv("VV_BINANCE_TEST_VENUE", "bogus")
    with _pytest.raises(ValueError):
        CcxtPerpClient("testnet")
