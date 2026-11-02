from datetime import datetime, timezone

import pytest

from vv.core.ids import client_id
from vv.core.orders import anchor_bucket, order_id, reconcile
from vv.core.types import (
    CancelOrder,
    OpenOrder,
    PlaceOrder,
    SymbolFilters,
    SymbolMarket,
)

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)
SYM = "BTC/USDT"
MKT = {SYM: SymbolMarket(SYM, TS, bid=100.0, ask=100.02, last=100.0, filters=FILT)}


def _open(side, price, qty, cid, filled=0.0, sym=SYM):
    return OpenOrder(sym, side, price, qty, filled, cid, "1", TS)


def _want(side, price, qty, cid="vv-new", post_only=True):
    return PlaceOrder(SYM, side, price, qty, cid, post_only)


def test_keeps_matching_order_within_tolerance():
    o = _open("buy", 99.0, 1.0, "vv-a")
    out = reconcile([_want("buy", 99.02, 1.05)], [o], markets=MKT)
    assert out == []


def test_replaces_when_price_beyond_tolerance_or_qty_off():
    o = _open("buy", 99.0, 1.0, "vv-a")
    far = reconcile([_want("buy", 99.03, 1.0)], [o], markets=MKT)
    assert far == [CancelOrder(SYM, "vv-a"), _want("buy", 99.03, 1.0)]
    big = reconcile([_want("buy", 99.0, 1.2)], [o], markets=MKT)
    assert big[0] == CancelOrder(SYM, "vv-a") and isinstance(big[1], PlaceOrder)


def test_ignores_non_vv_orders():
    foreign = _open("buy", 90.0, 1.0, "manual-1")
    out = reconcile([], [foreign], markets=MKT)
    assert out == []
    out = reconcile([_want("buy", 99.0, 1.0)], [foreign], markets=MKT)
    assert [type(i) for i in out] == [PlaceOrder]


def test_cancels_all_ours_when_nothing_desired_regime_blocks():
    orders = [_open("buy", 99.0, 1.0, "vv-a"), _open("buy", 98.0, 1.0, "vv-b")]
    out = reconcile([], orders, markets=MKT)
    assert out == [CancelOrder(SYM, "vv-a"), CancelOrder(SYM, "vv-b")]


def test_guard_drops_post_only_buy_at_or_above_bid_and_sell_at_or_below_bid():
    out = reconcile(
        [
            _want("buy", 100.0, 1.0, "vv-1"),
            _want("buy", 100.5, 1.0, "vv-2"),
            _want("sell", 100.0, 1.0, "vv-3"),
            _want("sell", 99.0, 1.0, "vv-4"),
        ],
        [],
        markets=MKT,
    )
    assert out == []


def test_guard_allows_non_post_only_sell_below_bid():
    out = reconcile([_want("sell", 99.0, 1.0, "vv-1", post_only=False)], [], markets=MKT)
    assert len(out) == 1


def test_guard_rounds_then_drops_below_min_notional():
    # 0.09 * 99 = 8.9 < 10 ; and qty rounds down to 0 below step
    out = reconcile(
        [_want("buy", 99.0, 0.09, "vv-1"), _want("buy", 99.0, 0.0004, "vv-2")],
        [],
        markets=MKT,
    )
    assert out == []


def test_rounds_prices_buy_down_sell_up():
    out = reconcile(
        [_want("buy", 99.019, 1.0, "vv-1"), _want("sell", 100.311, 1.0, "vv-2")],
        [],
        markets=MKT,
    )
    assert out[0].price == pytest.approx(99.01)
    assert out[1].price == pytest.approx(100.32)


def test_new_order_not_placed_with_id_of_kept_order():
    o = _open("buy", 99.0, 1.0, "vv-a")
    out = reconcile([_want("buy", 99.0, 1.0, "vv-z"), _want("buy", 97.0, 1.0, "vv-a")], [o], markets=MKT)
    assert out == []


def test_partially_filled_order_matches_on_remaining():
    o = _open("buy", 99.0, 2.0, "vv-a", filled=1.0)
    assert reconcile([_want("buy", 99.0, 1.0)], [o], markets=MKT) == []


def test_symbol_without_market_is_not_placed():
    out = reconcile([PlaceOrder("ETH/USDT", "buy", 10.0, 100.0, "vv-e")], [], markets=MKT)
    assert out == []


def test_anchor_bucket_stable_and_changes_with_anchor():
    assert anchor_bucket(100.0, 0.015) == anchor_bucket(100.0, 0.015)
    assert anchor_bucket(100.0, 0.015) == anchor_bucket(100.2, 0.015)
    assert anchor_bucket(100.0, 0.015) != anchor_bucket(110.0, 0.015)
    assert anchor_bucket(1e-9, 0.01) >= 0
    # client ids derived from it are stable
    a = client_id(SYM, "buy", 1, anchor_bucket(100.0, 0.015))
    assert a == client_id(SYM, "buy", 1, anchor_bucket(100.0, 0.015))


def test_order_id_unique_per_price_qty_and_deterministic():
    base = order_id(SYM, "sell", 0, 101.5, 1.0, 7)
    assert base == order_id(SYM, "sell", 0, 101.5, 1.0, 7)
    assert base != order_id(SYM, "sell", 0, 101.5, 1.2, 7)
    assert base != order_id(SYM, "sell", 0, 101.6, 1.0, 7)
    assert base.startswith("vv-") and len(base) <= 36


def test_order_id_changes_with_decision_minute_only():
    a = order_id(SYM, "buy", 1, 99.0, 1.0, 5, 1000)
    assert a == order_id(SYM, "buy", 1, 99.0, 1.0, 5, 1000)
    assert a != order_id(SYM, "buy", 1, 99.0, 1.0, 5, 1001)


def test_identical_order_is_kept_never_cancelled_and_replaced():
    cid = order_id(SYM, "buy", 1, 99.0, 1.0, 5)
    o = _open("buy", 99.0, 1.0, cid)
    assert reconcile([_want("buy", 99.0, 1.0, cid)], [o], markets=MKT) == []
