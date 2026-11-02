import random
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from vv.core.decide import decide
from vv.core.params import GridParams, RiskParams
from vv.core.types import (
    CancelOrder,
    Decision,
    HaltState,
    OpenOrder,
    PlaceOrder,
    PortfolioState,
    Position,
    Snapshot,
    SymbolFeatures,
    SymbolFilters,
    SymbolMarket,
)

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)
SYMS = ["A/USDT", "B/USDT", "C/USDT"]
GRID = GridParams(
    spacing_sigmas=1.5, min_step=0.004, sigma_horizon_minutes=60, levels=5,
    qty_multiplier=1.2, maker_fee=0.001,
)
RISK = RiskParams(
    total_exposure_cap=0.5, per_symbol_cap=0.3, portfolio_halt_dd=0.2, symbol_halt_dd=0.15
)


def _mkt(sym, bid=100.0):
    return SymbolMarket(sym, TS, bid=bid, ask=bid + 0.02, last=bid, filters=FILT)


def _feat(sigma=0.01, regime_ok=True, warm=True):
    return SymbolFeatures(close=100.0, sigma_h=sigma, ema_4h=100.0, regime_ok=regime_ok, warm=warm)


def _oo(sym, side, price, qty, cid):
    return OpenOrder(sym, side, price, qty, 0.0, cid, "1", TS)


def _snap(
    bids=None, positions=None, orders=(), cash=1000.0, peak=1000.0, halt=None, stale=(),
    features=None, symbols=SYMS,
):
    bids = bids or {}
    markets = {s: _mkt(s, bids.get(s, 100.0)) for s in symbols}
    pf = PortfolioState(
        ts=TS, cash_free=cash, cash_reserved=0.0, positions=positions or {},
        open_orders=tuple(orders), peak_equity=peak, halt=halt or HaltState(),
    )
    feats = features if features is not None else {s: _feat() for s in symbols}
    return Snapshot(ts=TS, markets=markets, bars={}, portfolio=pf, stale=stale, features=feats)


def _places(d: Decision, side=None, sym=None):
    return [
        i for i in d.intents
        if isinstance(i, PlaceOrder)
        and (side is None or i.side == side)
        and (sym is None or i.symbol == sym)
    ]


def _cancels(d: Decision):
    return [i for i in d.intents if isinstance(i, CancelOrder)]


def _apply(snap: Snapshot, d: Decision) -> list[OpenOrder]:
    """Orders resting after executing ``d`` (everything fills nowhere; just placed/cancelled)."""
    gone = {c.client_id for c in _cancels(d)}
    rest = [o for o in snap.portfolio.open_orders if o.client_id not in gone]
    rest += [_oo(p.symbol, p.side, p.price, p.qty, p.client_id) for p in _places(d)]
    return rest


def _buy_exposure(orders, positions, marks):
    total = sum(o.price * o.remaining for o in orders if o.side == "buy")
    return total + sum(p.qty * marks[p.symbol] for p in positions.values())


# --- basics ----------------------------------------------------------------


def test_flat_portfolio_gets_buy_ladder_below_bid_and_no_sells():
    d = decide(_snap(symbols=["A/USDT"]), GRID, RISK)
    buys = _places(d, "buy")
    assert len(buys) == GRID.levels
    assert all(b.price < 100.0 and b.post_only for b in buys)
    assert [b.price for b in buys] == sorted((b.price for b in buys), reverse=True)
    assert not _places(d, "sell") and not _cancels(d)
    assert d.halt == HaltState()
    assert isinstance(d.intents, tuple)


def test_determinism_same_snapshot_same_intents():
    snap = _snap(positions={"A/USDT": Position("A/USDT", 0.5, 101.0, 0.0)})
    assert decide(snap, GRID, RISK) == decide(snap, GRID, RISK)


def test_inputs_not_mutated():
    snap = _snap(positions={"A/USDT": Position("A/USDT", 0.5, 101.0, 0.0)})
    before = (dict(snap.portfolio.positions), snap.portfolio.open_orders, dict(snap.features))
    decide(snap, GRID, RISK)
    assert before == (dict(snap.portfolio.positions), snap.portfolio.open_orders, dict(snap.features))


# --- C1 ----------------------------------------------------------------------


def test_c1_no_post_only_buy_at_or_above_bid_randomized():
    rng = random.Random(12345)
    for _ in range(200):
        bids = {s: rng.uniform(0.5, 500.0) for s in SYMS}
        positions = {}
        for s in SYMS:
            if rng.random() < 0.6:
                positions[s] = Position(s, rng.uniform(0.01, 5.0), bids[s] * rng.uniform(0.7, 1.4), 0.0)
        feats = {
            s: _feat(rng.uniform(0.001, 0.05), regime_ok=rng.random() < 0.8, warm=rng.random() < 0.9)
            for s in SYMS
        }
        snap = _snap(bids=bids, positions=positions, features=feats, cash=rng.uniform(100, 5000),
                     peak=0.0)
        d = decide(snap, GRID, RISK)
        for p in _places(d):
            if p.post_only and p.side == "buy":
                assert p.price < snap.markets[p.symbol].bid
            if p.post_only and p.side == "sell":
                assert p.price > snap.markets[p.symbol].bid


# --- C2 ----------------------------------------------------------------------


def test_c2_symbol_halt_flattens_only_tripped_symbol_and_persists():
    pos = {
        "A/USDT": Position("A/USDT", 1.0, 100.0, 0.0),
        "B/USDT": Position("B/USDT", 1.0, 100.0, 0.0),
    }
    orders = [_oo("A/USDT", "buy", 80.0, 1.0, "vv-a"), _oo("B/USDT", "sell", 105.0, 1.0, "vv-b")]
    snap = _snap(bids={"A/USDT": 84.0}, positions=pos, orders=orders, cash=800.0, symbols=SYMS[:2])
    d = decide(snap, GRID, RISK)
    assert d.halt.active is False and d.halt.blocked_symbols == frozenset({"A/USDT"})
    assert CancelOrder("A/USDT", "vv-a") in d.intents
    sells = _places(d, "sell", "A/USDT")
    assert len(sells) == 1 and not sells[0].post_only and sells[0].qty == pytest.approx(1.0)
    assert sells[0].price < 84.0
    assert not _places(d, "buy", "A/USDT")
    # B keeps trading normally: take-profit and ladder placed
    assert _places(d, "sell", "B/USDT")
    assert _places(d, "buy", "B/USDT")

    # next cycle, A recovered: still blocked, still no buys on A
    snap2 = _snap(bids={"A/USDT": 100.0}, positions=pos, halt=d.halt, cash=800.0,
                  symbols=SYMS[:2])
    d2 = decide(snap2, GRID, RISK)
    assert d2.halt.blocked_symbols == frozenset({"A/USDT"})
    assert not _places(d2, "buy", "A/USDT")
    assert _places(d2, "sell", "A/USDT") and _places(d2, "buy", "B/USDT")


def test_c2_portfolio_halt_trips_flattens_everything_and_stays_latched():
    pos = {s: Position(s, 3.0, 100.0, 0.0) for s in SYMS[:2]}
    orders = [_oo("A/USDT", "buy", 80.0, 1.0, "vv-a"), _oo("C/USDT", "buy", 80.0, 1.0, "vv-c")]
    # equity = 400 + 6*65 = 790 < 800; symbols 65 < 85 as well
    snap = _snap(bids={s: 65.0 for s in SYMS}, positions=pos, orders=orders, cash=400.0)
    d = decide(snap, GRID, RISK)
    assert d.halt.active and d.halt.since == TS
    assert {c.client_id for c in _cancels(d)} == {"vv-a", "vv-c"}
    assert {p.symbol for p in _places(d)} == set(SYMS[:2])
    assert all(p.side == "sell" and not p.post_only for p in _places(d))

    # prices fully recovered next cycle: still halted, still flattening, no buys
    snap2 = _snap(bids={s: 100.0 for s in SYMS}, positions=pos, cash=400.0, halt=d.halt)
    d2 = decide(snap2, GRID, RISK)
    assert d2.halt == d.halt
    assert not _places(d2, "buy") and _places(d2, "sell")


# --- C3 ----------------------------------------------------------------------


def test_c3_own_resting_ladder_is_stable_not_cancelled_or_duplicated():
    snap = _snap(symbols=["A/USDT"])
    d = decide(snap, GRID, RISK)
    resting = _apply(snap, d)
    snap2 = _snap(symbols=["A/USDT"], orders=resting)
    assert decide(snap2, GRID, RISK).intents == ()


def test_c3_foreign_resting_buys_consuming_budget_prevent_new_buys():
    # budget for a lone symbol = 0.3 * 1000 = 300; a manual resting buy uses all of it
    foreign = [_oo("A/USDT", "buy", 90.0, 3.4, "manual-1")]  # 306 notional
    d = decide(_snap(symbols=["A/USDT"], orders=foreign), GRID, RISK)
    assert not _places(d, "buy")
    assert foreign[0].client_id not in {c.client_id for c in _cancels(d)}


def test_c3_holdings_consuming_budget_prevent_new_buys():
    pos = {"A/USDT": Position("A/USDT", 3.2, 100.0, 0.0)}  # 320 at mark > 300 budget
    d = decide(_snap(symbols=["A/USDT"], positions=pos, cash=680.0), GRID, RISK)
    assert not _places(d, "buy")


def test_c3_total_exposure_never_exceeds_cap_across_three_symbols():
    # A already holds 400 (budget 166 each, so A gets nothing), and the global cap of
    # 500 leaves only 100 for B and C together even though each has a 166 budget.
    pos = {"A/USDT": Position("A/USDT", 4.0, 100.0, 0.0)}
    snap = _snap(positions=pos, cash=600.0)
    d = decide(snap, GRID, RISK)
    marks = {s: (m.bid + m.ask) / 2 for s, m in snap.markets.items()}
    resting = _apply(snap, d)
    assert _buy_exposure(resting, pos, marks) <= 0.5 * 1000 + 1e-6
    assert not _places(d, "buy", "A/USDT")
    assert _places(d, "buy", "B/USDT") or _places(d, "buy", "C/USDT")


def test_c3_cap_holds_over_randomized_three_symbol_books():
    rng = random.Random(7)
    for _ in range(100):
        bids = {s: rng.uniform(1.0, 300.0) for s in SYMS}
        pos = {
            s: Position(s, rng.uniform(0.0, 4.0), bids[s] * rng.uniform(0.9, 1.3), 0.0)
            for s in SYMS if rng.random() < 0.5
        }
        orders = [
            _oo(s, "buy", bids[s] * 0.9, rng.uniform(0.1, 2.0), rng.choice(["vv-x" + s, "man" + s]))
            for s in SYMS if rng.random() < 0.4
        ]
        snap = _snap(bids=bids, positions=pos, orders=orders, cash=1000.0, peak=0.0)
        d = decide(snap, GRID, RISK)
        marks = {s: (m.bid + m.ask) / 2 for s, m in snap.markets.items()}
        before = _buy_exposure(snap.portfolio.open_orders, pos, marks)
        after = _buy_exposure(_apply(snap, d), pos, marks)
        eq = snap.portfolio.equity(marks)
        cap = 0.5 * eq
        # either within cap, or we added nothing and did not make it worse
        assert after <= cap + 1e-6 or after <= before + 1e-6


# --- C4 ----------------------------------------------------------------------


def test_c4_take_profit_never_below_cost_plus_fee_multiple_after_falling_fill():
    # Ladder filled on the way down: avg_entry 100, market now far below.
    pos = {"A/USDT": Position("A/USDT", 1.0, 100.0, 0.0)}
    d = decide(_snap(bids={"A/USDT": 92.0}, positions=pos, cash=900.0, symbols=["A/USDT"]),
               GRID, RISK)
    (tp,) = _places(d, "sell")
    assert tp.post_only and tp.qty == pytest.approx(1.0)
    assert tp.price >= 100.0 * (1 + 2.5 * GRID.round_trip_fee)
    assert tp.price > 92.02  # and above ask


def test_tp_uses_only_free_quantity():
    pos = {"A/USDT": Position("A/USDT", 1.0, 100.0, 0.0)}
    foreign = [_oo("A/USDT", "sell", 150.0, 0.4, "manual-s")]
    d = decide(_snap(positions=pos, orders=foreign, cash=900.0, symbols=["A/USDT"]), GRID, RISK)
    (tp,) = _places(d, "sell")
    assert tp.qty == pytest.approx(0.6)


# --- C5 ----------------------------------------------------------------------


def test_c5_stale_snapshot_gives_no_intents_even_with_halt_condition():
    pos = {s: Position(s, 3.0, 100.0, 0.0) for s in SYMS[:2]}
    snap = _snap(bids={s: 65.0 for s in SYMS}, positions=pos, cash=400.0, stale=("B/USDT",),
                 orders=[_oo("A/USDT", "buy", 80.0, 1.0, "vv-a")])
    d = decide(snap, GRID, RISK)
    assert d.intents == ()
    assert d.halt == snap.portfolio.halt


# --- regime / warm -----------------------------------------------------------


def test_regime_block_cancels_resting_buys_but_keeps_take_profit():
    pos = {"A/USDT": Position("A/USDT", 1.0, 100.0, 0.0)}
    base = _snap(positions=pos, cash=900.0, symbols=["A/USDT"])
    d0 = decide(base, GRID, RISK)
    resting = _apply(base, d0)
    blocked = _snap(positions=pos, cash=900.0, symbols=["A/USDT"], orders=resting,
                    features={"A/USDT": _feat(regime_ok=False)})
    d = decide(blocked, GRID, RISK)
    buy_ids = {o.client_id for o in resting if o.side == "buy"}
    assert buy_ids and {c.client_id for c in _cancels(d)} == buy_ids
    assert not _places(d)


def test_cold_symbol_buys_cancelled_sells_kept_and_nothing_new():
    pos = {"A/USDT": Position("A/USDT", 1.0, 100.0, 0.0)}
    orders = [_oo("A/USDT", "buy", 95.0, 1.0, "vv-b"), _oo("A/USDT", "sell", 110.0, 1.0, "vv-s")]
    snap = _snap(positions=pos, orders=orders, cash=900.0, symbols=["A/USDT"],
                 features={"A/USDT": _feat(warm=False)})
    d = decide(snap, GRID, RISK)
    assert d.intents == (CancelOrder("A/USDT", "vv-b"),)


def test_symbol_missing_features_is_skipped():
    d = decide(_snap(symbols=["A/USDT"], features={}), GRID, RISK)
    assert d.intents == ()


def test_tp_resized_after_new_fill_gets_a_different_client_id():
    snap1 = _snap(positions={"A/USDT": Position("A/USDT", 1.0, 100.0, 0.0)}, cash=900.0,
                  symbols=["A/USDT"])
    (tp1,) = _places(decide(snap1, GRID, RISK), "sell")
    resting = [_oo("A/USDT", "sell", tp1.price, tp1.qty, tp1.client_id)]
    # a new buy fill grows the position and moves the cost basis
    snap2 = _snap(positions={"A/USDT": Position("A/USDT", 1.5, 99.0, 0.0)}, cash=850.0,
                  orders=resting, symbols=["A/USDT"])
    d2 = decide(snap2, GRID, RISK)
    (tp2,) = _places(d2, "sell")
    assert tp2.client_id != tp1.client_id and tp2.qty == pytest.approx(1.5)
    assert CancelOrder("A/USDT", tp1.client_id) in d2.intents


def test_expired_flatten_is_retried_every_decide_with_new_id():
    pos = {s: Position(s, 3.0, 100.0, 0.0) for s in SYMS[:2]}
    snap = _snap(bids={s: 65.0 for s in SYMS}, positions=pos, cash=400.0)
    d1 = decide(snap, GRID, RISK)
    later = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    snap2 = Snapshot(later, snap.markets, {}, _snap(
        bids={s: 65.0 for s in SYMS}, positions=pos, cash=400.0, halt=d1.halt).portfolio,
        features=snap.features)
    d2 = decide(snap2, GRID, RISK)
    ids1 = {p.client_id for p in _places(d1, "sell")}
    ids2 = {p.client_id for p in _places(d2, "sell")}
    assert ids1 and ids2 and ids1.isdisjoint(ids2)
    assert all(p.price == pytest.approx(65.0 * 0.99, abs=0.01) for p in _places(d2, "sell"))


# --- S1.5: ladder hysteresis and per-minute ids ---------------------------------------

STEP = 0.015  # 1.5 * sigma 0.01, above GRID.min_step


def _run_cycles(bids):
    """Feed successive flat snapshots (one per decision minute), applying each decision."""
    resting: list[OpenOrder] = []
    out = []
    for k, bid in enumerate(bids):
        base = _snap(bids={"A/USDT": bid}, orders=resting, symbols=["A/USDT"])
        ts = datetime(2026, 1, 1, 0, k, tzinfo=timezone.utc)
        snap = replace(base, ts=ts)
        d = decide(snap, GRID, RISK)
        out.append(d)
        resting = _apply(snap, d)
    return out


def test_s15_flat_bid_drifting_within_half_step_causes_no_churn():
    # 0.3 step up, back, up, back: always inside the +-0.5 step hysteresis band
    bids = [100.0 * (1 + 0.3 * STEP * (k % 2)) for k in range(5)]
    ds = _run_cycles(bids)
    assert len(_places(ds[0], "buy")) == GRID.levels
    for d in ds[1:]:
        assert d.intents == ()


def test_s15_bid_rising_0p6_step_replaces_the_ladder():
    ds = _run_cycles([100.0, 100.0 * (1 + 0.6 * STEP)])
    assert len(_cancels(ds[1])) == GRID.levels
    assert len(_places(ds[1], "buy")) == GRID.levels


def test_s15_bid_falling_0p6_step_replaces_the_ladder():
    ds = _run_cycles([100.0, 100.0 * (1 - 0.6 * STEP)])
    assert len(_cancels(ds[1])) == GRID.levels
    assert len(_places(ds[1], "buy")) == GRID.levels


def test_s15_position_keeps_anchor_behaviour_not_hysteresis():
    # held: anchor = min(bid, avg_entry); a bid far above entry changes nothing, a bid
    # below entry re-anchors the ladder
    pos = {"A/USDT": Position("A/USDT", 1.0, 100.0, 0.0)}
    snap = _snap(positions=pos, cash=900.0, symbols=["A/USDT"])
    resting = [o for o in _apply(snap, decide(snap, GRID, RISK))]
    up = _snap(bids={"A/USDT": 103.0}, positions=pos, cash=900.0, orders=resting,
               symbols=["A/USDT"])
    d_up = decide(up, GRID, RISK)  # only the take-profit follows the ask; buys untouched
    assert not _places(d_up, "buy")
    assert all(c.client_id not in {o.client_id for o in resting if o.side == "buy"}
               for c in _cancels(d_up))
    down = _snap(bids={"A/USDT": 97.0}, positions=pos, cash=900.0, orders=resting,
                 symbols=["A/USDT"])
    assert _cancels(decide(down, GRID, RISK))


def test_s15_same_order_gets_different_id_in_a_different_minute():
    snap = _snap(symbols=["A/USDT"])
    later = replace(snap, ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    same_minute = replace(snap, ts=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc))
    a, b, c = (decide(s, GRID, RISK) for s in (snap, later, same_minute))
    ids = lambda d: [p.client_id for p in _places(d, "buy")]  # noqa: E731
    assert [(p.price, p.qty) for p in _places(a)] == [(p.price, p.qty) for p in _places(b)]
    assert set(ids(a)).isdisjoint(ids(b))
    assert ids(a) == ids(c)


def test_s15_sell_id_also_changes_with_the_minute():
    pos = {"A/USDT": Position("A/USDT", 1.0, 100.0, 0.0)}
    snap = _snap(positions=pos, cash=900.0, symbols=["A/USDT"])
    later = replace(snap, ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    (s1,) = _places(decide(snap, GRID, RISK), "sell")
    (s2,) = _places(decide(later, GRID, RISK), "sell")
    assert (s1.price, s1.qty) == (s2.price, s2.qty) and s1.client_id != s2.client_id
