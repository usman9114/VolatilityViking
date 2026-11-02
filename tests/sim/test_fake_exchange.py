"""FakeExchange: multi-coin, shared wallet, pessimistic fills.

Bars are labelled by open time; at virtual time t the newest visible bar is the
one labelled t-1m, and an order placed at t first meets the bar labelled t.
Test layout: bar 0 is a flat "setup" bar; ``prime`` advances past it so symbols
are listed, then orders are placed and ``step`` runs the following bars.
"""

from datetime import timedelta, timezone

import pandas as pd
import pytest

from vv.core.ports import Clock, ExchangeAdapter
from vv.core.types import CancelOrder, PlaceOrder, SymbolFilters
from vv.sim.fake_exchange import FakeExchange, OrderRejected

T0 = pd.Timestamp("2026-01-01", tz="UTC")
MIN = timedelta(minutes=1)
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)
A, B = "AAA/USDT", "BBB/USDT"


def bars(rows, start=T0):
    """rows: list of (open, high, low, close); one bar per minute from ``start``."""
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


def flat(px):
    return (px, px, px, px)


def make(rows_by_sym, **kw):
    kw.setdefault("start_cash", 1000.0)
    kw.setdefault("maker_fee", 0.001)
    kw.setdefault("taker_fee", 0.001)
    kw.setdefault("taker_slippage_bps", 0)
    syms = list(rows_by_sym)
    data = {s: r if isinstance(r, pd.DataFrame) else bars(r) for s, r in rows_by_sym.items()}
    return FakeExchange(data, {s: FILT for s in syms}, **kw)


def buy(cid, qty, price, sym=A, post_only=True):
    return PlaceOrder(sym, "buy", price, qty, cid, post_only)


def sell(cid, qty, price, sym=A, post_only=True):
    return PlaceOrder(sym, "sell", price, qty, cid, post_only)


def at(n):
    return T0 + n * MIN


async def reason(coro):
    with pytest.raises(OrderRejected) as e:
        await coro
    return e.value.reason


def test_conforms_to_ports():
    ex = make({A: [flat(100)] * 3})
    assert isinstance(ex, ExchangeAdapter)
    assert isinstance(ex.clock, Clock)
    assert ex.clock.now().tzinfo is timezone.utc or ex.clock.now().utcoffset() == timedelta(0)


# 1 -------------------------------------------------------------- touch != fill
async def test_touch_is_not_fill_but_trade_through_is():
    ex = make({A: [flat(100), (100, 100, 98.99, 99.5), (99.5, 99.5, 98.98, 99), flat(99)]})
    ex.advance_to(at(1))
    await ex.create_order(buy("b1", 1, 99.00))
    ex.advance_to(at(2))  # bar 1: low == price - tick -> touch only
    assert ex.fills == []
    assert len(await ex.fetch_open_orders()) == 1
    ex.advance_to(at(3))  # bar 2: low < price - tick -> trade-through
    assert [f["client_id"] for f in ex.fills] == ["b1"]
    assert ex.fills[0]["price"] == 99.00
    assert await ex.fetch_open_orders() == []


async def test_sell_needs_high_through_price_plus_tick():
    rows = [flat(100), (100, 100, 100, 100), (100, 101.01, 100, 101), (101, 101.02, 101, 101)]
    ex = make({A: rows})
    ex.advance_to(at(1))
    await ex.create_order(buy("b", 1, 100.00, post_only=False))  # taker fills at bar 1 open
    ex.advance_to(at(2))
    await ex.create_order(sell("s", 1, 101.00))
    ex.advance_to(at(3))  # bar 2: high == price + tick -> touch
    assert [f["client_id"] for f in ex.fills] == ["b"]
    ex.advance_to(at(4))  # bar 3: high 101.02 > 101.01 -> fill
    assert [f["client_id"] for f in ex.fills] == ["b", "s"]


# 2 --------------------------------------------------------- post-only crossing
async def test_post_only_crossing_rejected():
    ex = make({A: [flat(100)] * 4})
    ex.advance_to(at(1))
    # ask = 100.005, bid = 99.995
    assert await reason(ex.create_order(buy("b", 1, 100.01))) == "post_only_cross"
    await ex.create_order(buy("b-ok", 1, 100.00))
    assert await reason(ex.create_order(buy("b2", 1, 105.00))) == "post_only_cross"
    assert len(await ex.fetch_open_orders()) == 1  # nothing was filled as taker
    assert ex.fills == []


async def test_post_only_sell_crossing_rejected():
    ex = make({A: [flat(100), (100, 100, 100, 100), (100, 100, 99, 99.5), flat(100), flat(100)]})
    ex.advance_to(at(1))
    await ex.create_order(buy("b", 1, 100.00, post_only=False))
    ex.advance_to(at(2))
    assert ex._pos[A].qty == 1
    assert await reason(ex.create_order(sell("s", 1, 99.99))) == "post_only_cross"
    assert await reason(ex.create_order(sell("s2", 1, 90.00))) == "post_only_cross"


# 3 ------------------------------------------------------------- cash reservation
async def test_cash_reserved_at_placement_and_released_on_cancel():
    ex = make({A: [flat(100)] * 4})
    ex.advance_to(at(1))
    await ex.create_order(buy("b1", 5, 100.00))  # 500 + 0.1% fee
    st = ex.portfolio_state(peak_equity=1000.0)
    assert st.cash_reserved == pytest.approx(500.5)
    assert st.cash_free == pytest.approx(499.5)
    assert st.equity({}) == pytest.approx(1000.0)
    assert await reason(ex.create_order(buy("b2", 6, 100.00))) == "insufficient_cash"
    await ex.cancel_order(CancelOrder(A, "b1"))
    st = ex.portfolio_state(peak_equity=1000.0)
    assert st.cash_reserved == 0 and st.cash_free == pytest.approx(1000.0)
    await ex.create_order(buy("b3", 9, 100.00))  # now affordable
    assert ex.portfolio_state(1000.0).cash_reserved == pytest.approx(900.9)


async def test_cancel_unknown_order_rejected():
    ex = make({A: [flat(100)] * 3})
    ex.advance_to(at(1))
    assert await reason(ex.cancel_order(CancelOrder(A, "nope"))) == "unknown_order"


# 4 ------------------------------------------------------------- shared wallet
async def test_two_symbols_compete_for_one_wallet():
    ex = make({A: [flat(100)] * 4, B: [flat(50)] * 4})
    ex.advance_to(at(1))
    await ex.create_order(buy("a", 8, 100.00, A))  # 800.8
    assert await reason(ex.create_order(buy("b", 5, 50.00, B))) == "insufficient_cash"  # 250.25 > 199.2
    await ex.create_order(buy("b-small", 3, 50.00, B))  # 150.15 fits
    st = ex.portfolio_state(1000.0)
    assert st.cash_reserved == pytest.approx(800.8 + 150.15)
    assert len(st.open_orders) == 2
    bal = await ex.fetch_balance()
    assert bal["USDT"] == pytest.approx(1000 - 800.8 - 150.15)


async def test_unlisted_symbol_rejected_until_it_has_bars():
    late = bars([flat(50)] * 4, start=at(5))
    ex = make({A: [flat(100)] * 10, B: late})
    ex.advance_to(at(2))
    assert await reason(ex.create_order(buy("b", 1, 50.00, B))) == "not_listed"
    assert set(await ex.fetch_tickers([A, B])) == {A}
    ex.advance_to(at(6))  # B's first bar (labelled 5) has closed
    assert set(await ex.fetch_tickers([A, B])) == {A, B}
    await ex.create_order(buy("b", 1, 50.00, B))
    assert await reason(ex.create_order(buy("x", 1, 50.00, "ZZZ/USDT"))) == "unknown_symbol"


# 5 --------------------------------------------------------------- bar timing
async def test_no_same_bar_fill():
    # bar 0 already pierces 95; an order placed after it must not fill from it
    ex = make({A: [(100, 100, 90, 100), (100, 100, 100, 100), (100, 100, 94, 95), flat(95)]})
    ex.advance_to(at(1))
    await ex.create_order(buy("b", 1, 95.00))
    assert ex.fills == []
    ex.advance_to(at(2))  # bar 1 flat: still nothing
    assert ex.fills == []
    ex.advance_to(at(3))  # bar 2 trades through
    assert len(ex.fills) == 1


async def test_order_placed_mid_bar_cannot_fill_that_bar():
    ex = make({A: [flat(100), (100, 100, 90, 95), (95, 95, 85, 90), flat(90)]})
    ex.advance_to(at(1) + timedelta(seconds=30))  # inside bar 1
    await ex.create_order(buy("b", 1, 95.00))
    ex.advance_to(at(2))  # bar 1 closes: it was already open when we placed
    assert ex.fills == []
    ex.advance_to(at(3))
    assert len(ex.fills) == 1


@pytest.mark.parametrize(
    "bar, first",
    [
        ((100, 106, 94, 95), "sell"),  # down bar: O-H-L-C -> sell first
        ((100, 106, 94, 105), "buy"),  # up bar:   O-L-H-C -> buy first
    ],
)
async def test_intrabar_path_order(bar, first):
    ex = make({A: [flat(100), (100, 100, 99.5, 100), bar, flat(100)]})
    ex.advance_to(at(1))
    await ex.create_order(buy("inv", 1, 100.00, post_only=False))
    ex.advance_to(at(2))
    assert ex._pos[A].qty == 1
    ex.fills.clear()
    # resting at the same time: sell above and buy below the market
    await ex.create_order(sell("s", 1, 103.00))
    await ex.create_order(buy("b", 1, 97.00))
    ex.advance_to(at(3))
    assert [f["side"] for f in ex.fills] == [first, "buy" if first == "sell" else "sell"]


# 6 -------------------------------------------------------------- fee maths
async def test_fees_and_avg_entry_three_buys_one_sell():
    rows = [
        flat(100),
        (100, 100, 98, 99),  # bar 1: fills b1 @100? limit 100 -> needs low < 99.99
        (99, 99, 88, 89),  # bar 2: b2 @ 90 (low 88 < 89.99)
        (89, 89, 78, 79),  # bar 3: b3 @ 80 (low 78 < 79.99)
        (79, 112, 79, 111),  # bar 4: sell 2 @ 110 (high 112 > 110.01)
        flat(111),
    ]
    ex = make({A: rows})
    ex.advance_to(at(1))
    await ex.create_order(buy("b1", 1, 100.00))  # ask is 100.005 -> ok at 100.00
    ex.advance_to(at(2))
    await ex.create_order(buy("b2", 2, 90.00))
    ex.advance_to(at(3))
    await ex.create_order(buy("b3", 1, 80.00))
    ex.advance_to(at(4))
    assert [f["client_id"] for f in ex.fills] == ["b1", "b2", "b3"]
    pos = ex.portfolio_state(1000.0).positions[A]
    # cost incl. fees: 100.1 + 180.18 + 80.08 = 360.36 over 4 units
    assert pos.qty == pytest.approx(4)
    assert pos.avg_entry == pytest.approx(360.36 / 4)  # 90.09
    assert ex.cash_free == pytest.approx(1000 - 360.36)
    assert [f["fee"] for f in ex.fills] == pytest.approx([0.1, 0.18, 0.08])

    await ex.create_order(sell("s1", 2, 110.00))
    ex.advance_to(at(5))
    st = ex.portfolio_state(1000.0)
    pos = st.positions[A]
    # proceeds 220, fee 0.22; realized = 219.78 - 2 * 90.09 = 39.60
    assert ex.fills[-1]["fee"] == pytest.approx(0.22)
    assert pos.qty == pytest.approx(2)
    assert pos.avg_entry == pytest.approx(90.09)
    assert pos.realized_pnl == pytest.approx(39.60)
    assert st.cash_free == pytest.approx(1000 - 360.36 + 219.78)
    assert st.cash_reserved == 0
    assert st.equity({A: 111.0}) == pytest.approx(1000 - 360.36 + 219.78 + 2 * 111)


async def test_full_close_resets_avg_entry_keeps_realized():
    rows = [flat(100), (100, 100, 98, 99), (99, 103, 99, 103), flat(103)]
    ex = make({A: rows})
    ex.advance_to(at(1))
    await ex.create_order(buy("b", 1, 100.00))
    ex.advance_to(at(2))
    await ex.create_order(sell("s", 1, 102.00))
    ex.advance_to(at(3))
    pos = ex.portfolio_state(1000.0).positions[A]
    assert pos.qty == 0 and pos.avg_entry == 0
    assert pos.realized_pnl == pytest.approx((102 - 0.102) - 100.1)
    assert (await ex.fetch_balance()).keys() == {"USDT"}


# 7 -------------------------------------------------------------- taker flatten
async def test_taker_flatten_fills_at_next_open_with_slippage():
    rows = [flat(100), (100, 100, 99.5, 100), (100, 101, 100, 100.5), (90, 91, 89, 90), flat(90)]
    ex = make({A: rows}, taker_slippage_bps=10)
    ex.advance_to(at(1))
    await ex.create_order(buy("inv", 1, 101.00, post_only=False))
    ex.advance_to(at(2))
    # bar 1 open 100 -> buy at 100.1 (slippage up), taker fee 0.1%
    f = ex.fills[0]
    assert f["price"] == pytest.approx(100.10)
    assert f["fee"] == pytest.approx(100.10 * 0.001)
    assert f["liquidity"] == "taker"
    assert ex.fills[0]["ts"] == at(2)

    ex.advance_to(at(3))  # bar 2 closed; bar 3 will open at 90 (a gap down)
    await ex.create_order(sell("flat", 1, 50.00, post_only=False))
    assert len(await ex.fetch_open_orders()) == 1  # not filled at placement
    ex.advance_to(at(4))  # bar 3 opens 90 -> sell at 90 * 0.999 = 89.91
    f = ex.fills[-1]
    assert f["side"] == "sell" and f["client_id"] == "flat"
    assert f["price"] == pytest.approx(89.91)
    assert f["fee"] == pytest.approx(89.91 * 0.001)
    assert ex.portfolio_state(1000.0).positions[A].qty == 0


async def test_taker_beyond_limit_expires_and_releases_cash():
    rows = [flat(100), flat(100), (120, 121, 119, 120), flat(120)]
    ex = make({A: rows})
    ex.advance_to(at(2))
    await ex.create_order(buy("chase", 1, 105.00, post_only=False))  # reserves 105.105
    assert ex.cash_reserved == pytest.approx(105.105)
    ex.advance_to(at(3))  # bar 2 opens at 120 > limit -> IOC expires
    assert ex.fills == [] and len(ex.expired) == 1
    assert await ex.fetch_open_orders() == []
    assert ex.cash_free == pytest.approx(1000.0)


# 8 -------------------------------------------------------------- idempotency
async def test_duplicate_client_id_rejected():
    ex = make({A: [flat(100)] * 5})
    ex.advance_to(at(1))
    await ex.create_order(buy("dup", 1, 99.00))
    assert await reason(ex.create_order(buy("dup", 1, 98.00))) == "duplicate_client_id"
    await ex.cancel_order(CancelOrder(A, "dup"))
    assert await reason(ex.create_order(buy("dup", 1, 98.00))) == "duplicate_client_id"
    assert await ex.fetch_open_orders() == []


# 9 -------------------------------------------------------------- no look-ahead
async def test_fetch_ohlcv_never_returns_future_bars():
    rows = [(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(30)]
    ex = make({A: rows})
    for n in (0, 1, 7, 15, 29, 30):
        ex.advance_to(at(n))
        df = await ex.fetch_ohlcv(A, "1m", 1000)
        assert len(df) == n  # bars labelled < now only
        if n:
            assert df.index[-1] == at(n - 1)
            assert df.index[-1] + MIN <= pd.Timestamp(ex.clock.now())
    ex = make({A: rows})
    ex.advance_to(at(12) + timedelta(seconds=30))
    df = await ex.fetch_ohlcv(A, "1m", 5)
    assert list(df.index) == [at(i) for i in range(7, 12)]  # bar 12 still open
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    since = await ex.fetch_ohlcv(A, "1m", 100, since=at(10).to_pydatetime())
    assert list(since.index) == [at(10), at(11)]


async def test_fetch_ohlcv_resamples_closed_buckets_only():
    rows = [(i, i + 1, i - 1, i + 0.5) for i in range(1, 21)]
    ex = make({A: rows})
    ex.advance_to(at(13))  # 5m bucket [10,15) not complete
    df = await ex.fetch_ohlcv(A, "5m", 10)
    assert list(df.index) == [at(0), at(5)]
    first = df.iloc[0]
    assert (first.open, first.high, first.low, first.close) == (1, 6, 0, 5.5)
    assert first.volume == 5
    ex.advance_to(at(15))
    assert list((await ex.fetch_ohlcv(A, "5m", 10)).index) == [at(0), at(5), at(10)]


async def test_tickers_use_last_closed_bar_with_one_tick_spread():
    ex = make({A: [flat(100), flat(101), flat(102)]})
    ex.advance_to(at(2))
    t = (await ex.fetch_tickers([A]))[A]
    assert t.last == 101  # bar labelled 2 has not closed
    assert t.ask - t.bid == pytest.approx(FILT.tick_size)
    assert t.bid == pytest.approx(101 - 0.005) and t.ask == pytest.approx(101 + 0.005)
    assert t.filters == FILT and t.ts == ex.clock.now()


# 10 ------------------------------------------------------------- haircut / seed
def _grid_rows(n):
    rows = [flat(100)]
    for i in range(n):
        rows.append((100, 100, 90, 100) if i % 2 == 0 else (100, 110, 100, 100))
    return rows


async def _run(haircut, seed):
    ex = make({A: _grid_rows(40)}, fill_haircut=haircut, seed=seed, start_cash=100_000.0)
    ex.advance_to(at(1))
    await ex.create_order(buy("inv", 20, 100.00, post_only=False))
    ex.advance_to(at(2))
    ex.fills.clear()
    for i in range(2, 40):
        ex.advance_to(at(i))
        for cid, side, px in ((f"b{i}", "buy", 99.00), (f"s{i}", "sell", 101.00)):
            try:
                await ex.create_order(PlaceOrder(A, side, px, 1, cid))
            except OrderRejected:
                pass
    ex.advance_to(at(41))
    return [(f["ts"], f["client_id"]) for f in ex.fills]


async def test_haircut_zero_fills_nothing():
    assert await _run(0.0, 0) == []


async def test_haircut_full_fills_something_and_seed_is_deterministic():
    full = await _run(1.0, 0)
    assert len(full) > 0
    half_a, half_b = await _run(0.5, 7), await _run(0.5, 7)
    assert half_a == half_b
    # unfilled eligible orders keep resting, so a haircut delays fills rather than dropping them
    assert 0 < len(half_a) <= len(full)
    assert half_a != full
    assert sum(t.timestamp() for t, _ in half_a) / len(half_a) > sum(t.timestamp() for t, _ in full) / len(full)
    assert await _run(0.5, 8) != half_a


# validation ----------------------------------------------------------------
async def test_tick_step_and_min_notional_validation():
    ex = make({A: [flat(100)] * 3})
    ex.advance_to(at(1))
    assert await reason(ex.create_order(buy("a", 1, 99.005))) == "not_on_tick"
    assert await reason(ex.create_order(buy("b", 1.0005, 99.00))) == "not_on_step"
    assert await reason(ex.create_order(buy("c", 0.09, 99.00))) == "below_min_notional"
    assert await reason(ex.create_order(buy("d", 0.0, 99.00))) == "invalid_order"
    assert await reason(ex.create_order(sell("e", 1, 101.00))) == "insufficient_base"
    await ex.create_order(buy("ok", 0.102, 99.00))  # 10.098 >= min_notional


async def test_sells_need_free_base_across_resting_sells():
    rows = [flat(100), (100, 100, 99.5, 100), flat(100), flat(100)]
    ex = make({A: rows})
    ex.advance_to(at(1))
    await ex.create_order(buy("inv", 1, 100.00, post_only=False))
    ex.advance_to(at(2))
    await ex.create_order(sell("s1", 0.6, 105.00))
    assert await reason(ex.create_order(sell("s2", 0.6, 105.00))) == "insufficient_base"
    assert (await ex.fetch_balance()).get("AAA") == pytest.approx(0.4)
    await ex.create_order(sell("s3", 0.4, 105.00))


async def test_fetch_my_trades_filters_by_symbol_and_since():
    rows = [flat(100), (100, 100, 98, 99), (99, 99, 97, 98), flat(98)]
    ex = make({A: rows, B: [flat(50)] * 4})
    ex.advance_to(at(1))
    await ex.create_order(buy("a1", 1, 99.00))
    ex.advance_to(at(2))
    await ex.create_order(buy("a2", 1, 98.00))
    ex.advance_to(at(3))
    allt = await ex.fetch_my_trades(A)
    assert [t["client_id"] for t in allt] == ["a1", "a2"]
    assert set(allt[0]) >= {"ts", "symbol", "side", "price", "qty", "fee", "client_id"}
    late = await ex.fetch_my_trades(A, since=at(3).to_pydatetime())
    assert [t["client_id"] for t in late] == ["a2"]
    assert await ex.fetch_my_trades(B) == []


def test_clock_is_forward_only():
    ex = make({A: [flat(100)] * 3})
    ex.advance_to(at(2))
    with pytest.raises(ValueError):
        ex.advance_to(at(1))
    with pytest.raises(ValueError):
        ex.advance_to(pd.Timestamp("2026-01-01 00:05"))  # naive
