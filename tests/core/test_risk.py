from datetime import datetime, timezone

import pytest

from vv.core.params import RiskParams
from vv.core.risk import evaluate_halt, flatten_intents
from vv.core.types import (
    CancelOrder,
    HaltState,
    OpenOrder,
    PlaceOrder,
    PortfolioState,
    Position,
    SymbolFilters,
    SymbolMarket,
)

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)
RISK = RiskParams(
    total_exposure_cap=0.5, per_symbol_cap=0.3, portfolio_halt_dd=0.2, symbol_halt_dd=0.15
)


def _mkt(sym, bid):
    return SymbolMarket(sym, TS, bid=bid, ask=bid + 0.02, last=bid, filters=FILT)


def _oo(sym, side, price, qty, cid):
    return OpenOrder(sym, side, price, qty, 0.0, cid, "1", TS)


def _pf(cash=500.0, peak=1000.0, halt=None, orders=()):
    return PortfolioState(
        ts=TS,
        cash_free=cash,
        cash_reserved=0.0,
        positions={
            "A/USDT": Position("A/USDT", 2.0, 100.0, 0.0),  # 200 at entry
            "B/USDT": Position("B/USDT", 3.0, 100.0, 0.0),  # 300 at entry
        },
        open_orders=tuple(orders),
        peak_equity=peak,
        halt=halt or HaltState(),
    )


def test_no_trip_when_healthy():
    h = evaluate_halt(_pf(), {"A/USDT": 100.0, "B/USDT": 100.0}, RISK, NOW)
    assert h == HaltState()


def test_portfolio_trips_at_threshold_not_before():
    # equity = 500 + 5*m ; floor = 800 -> m < 60 trips
    ok = evaluate_halt(_pf(), {"A/USDT": 60.0, "B/USDT": 60.0}, RISK, NOW)
    assert ok.active is False
    tripped = evaluate_halt(_pf(), {"A/USDT": 59.9, "B/USDT": 59.9}, RISK, NOW)
    assert tripped.active and tripped.since == NOW and "portfolio" in tripped.reason


def test_symbol_trip_blocks_only_that_symbol():
    h = evaluate_halt(_pf(), {"A/USDT": 84.0, "B/USDT": 100.0}, RISK, NOW)
    assert h.active is False
    assert h.blocked_symbols == frozenset({"A/USDT"})
    exact = evaluate_halt(_pf(), {"A/USDT": 85.0, "B/USDT": 100.0}, RISK, NOW)
    assert exact.blocked_symbols == frozenset()  # mark == avg_entry*(1-dd) is not below


def test_halt_is_latched_after_recovery():
    marks = {"A/USDT": 100.0, "B/USDT": 100.0}
    prev = HaltState(True, "x", TS, frozenset({"A/USDT"}))
    h = evaluate_halt(_pf(halt=prev), marks, RISK, NOW)
    assert h == prev  # unchanged: reason/since preserved too


def test_blocked_symbol_latched_and_accumulates():
    marks = {"A/USDT": 100.0, "B/USDT": 80.0}
    prev = HaltState(False, "symbol drawdown: A/USDT", TS, frozenset({"A/USDT"}))
    h = evaluate_halt(_pf(halt=prev), marks, RISK, NOW)
    assert h.blocked_symbols == frozenset({"A/USDT", "B/USDT"})
    assert h.since == TS  # first trip time kept


def test_flatten_cancels_vv_orders_and_sells_free_qty_below_bid():
    orders = [
        _oo("A/USDT", "buy", 90.0, 1.0, "vv-1"),
        _oo("A/USDT", "sell", 120.0, 1.0, "vv-2"),
        _oo("A/USDT", "sell", 130.0, 0.5, "manual-1"),  # foreign: locks 0.5 of the position
    ]
    pf = _pf(orders=orders)
    out = flatten_intents(pf, {"A/USDT": _mkt("A/USDT", 80.0)}, ["A/USDT"], NOW)
    assert out[:2] == [CancelOrder("A/USDT", "vv-1"), CancelOrder("A/USDT", "vv-2")]
    sell = out[2]
    assert isinstance(sell, PlaceOrder) and sell.side == "sell" and sell.post_only is False
    assert sell.qty == pytest.approx(1.5)
    assert sell.price == pytest.approx(80.0 * 0.99, abs=0.01)
    assert sell.client_id.startswith("vv-") and len(out) == 3


def test_flatten_only_named_symbols():
    pf = _pf(orders=[_oo("B/USDT", "buy", 90.0, 1.0, "vv-b")])
    out = flatten_intents(pf, {"A/USDT": _mkt("A/USDT", 80.0)}, ["A/USDT"], NOW)
    assert all(i.symbol == "A/USDT" for i in out)


def test_flatten_keeps_matching_resting_flatten_order():
    mk = {"A/USDT": _mkt("A/USDT", 80.0)}
    first = flatten_intents(_pf(), mk, ["A/USDT"], NOW)
    sell = first[0]
    resting = _oo("A/USDT", "sell", sell.price, sell.qty, sell.client_id)
    again = flatten_intents(_pf(orders=[resting]), mk, ["A/USDT"], NOW)
    assert again == []


def test_flatten_dust_not_sold():
    pf = PortfolioState(
        TS, 0.0, 0.0, {"A/USDT": Position("A/USDT", 0.01, 100.0, 0.0)}, (), 1.0, HaltState()
    )
    assert flatten_intents(pf, {"A/USDT": _mkt("A/USDT", 100.0)}, ["A/USDT"], NOW) == []


def test_flatten_expired_ioc_is_reissued_with_new_client_id():
    mk = {"A/USDT": _mkt("A/USDT", 80.0)}
    t0 = NOW
    t1 = datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc)
    first = flatten_intents(_pf(), mk, ["A/USDT"], t0)[0]
    # IOC expired: nothing resting, position unchanged next minute -> retried
    retry = flatten_intents(_pf(), mk, ["A/USDT"], t1)
    assert len(retry) == 1 and isinstance(retry[0], PlaceOrder)
    assert retry[0].client_id != first.client_id
    assert retry[0].qty == first.qty and not retry[0].post_only
    # same minute, same inputs -> same id (deterministic)
    assert flatten_intents(_pf(), mk, ["A/USDT"], t0)[0].client_id == first.client_id
