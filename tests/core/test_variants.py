"""Variants B (paired grid) and C (rebalancing): pure decision logic on hand-built snapshots."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from vv.core import filters as flt
from vv.core.params import RiskParams
from vv.core.types import (
    CancelOrder,
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
from vv.core.variants.paired_grid import PairedGridParams, decide_paired
from vv.core.variants.rebalance import RebalanceParams, decide_rebalance

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)
SYM = "AAAUSDT"
RISK = RiskParams(
    total_exposure_cap=0.38, per_symbol_cap=0.15, portfolio_halt_dd=0.15, symbol_halt_dd=0.25
)
PARAMS = PairedGridParams(levels=10, step=0.02)
CASH = 10_000.0


def _feat(regime_ok=True, warm=True, sigma=0.01):
    return SymbolFeatures(close=100.0, sigma_h=sigma, ema_4h=100.0, regime_ok=regime_ok, warm=warm)


class Sim:
    """A tiny book keeper around a pure decision function: cash, positions, resting orders and a
    crude matcher (a resting buy fills when the bid trades below it, a sell when the ask is above)."""

    def __init__(self, symbols=(SYM,), risk=RISK, params=PARAMS):
        self.symbols, self.risk, self.params = tuple(symbols), risk, params
        self.state = None
        self.t = T0
        self.cash = CASH
        self.qty = {s: 0.0 for s in self.symbols}
        self.orders: list[OpenOrder] = []
        self.pending: list[dict] = []
        self.peak = CASH
        self.bids = {s: 100.0 for s in self.symbols}
        self.seq = 0

    def snapshot(self, minutes=15, stale=(), features=None, peak=None):
        self.t += timedelta(minutes=minutes)
        markets = {
            s: SymbolMarket(s, self.t, b, b + 0.02, b, FILT) for s, b in self.bids.items()
        }
        reserved = sum(o.price * o.remaining for o in self.orders if o.side == "buy")
        positions = {
            s: Position(s, q, 100.0, 0.0) for s, q in self.qty.items() if q > 0
        }
        pf = PortfolioState(
            ts=self.t, cash_free=self.cash - reserved, cash_reserved=reserved,
            positions=positions, open_orders=tuple(self.orders),
            peak_equity=self.peak if peak is None else peak, halt=HaltState(),
        )
        feats = features if features is not None else {s: _feat() for s in self.symbols}
        return Snapshot(
            ts=self.t, markets=markets, bars={}, portfolio=pf, stale=stale, features=feats,
            fills_since_last=tuple(self.pending),
        )

    def apply(self, decision):
        self.pending = []
        for i in decision.intents:
            if isinstance(i, CancelOrder):
                self.orders = [o for o in self.orders if o.client_id != i.client_id]
            else:
                self.seq += 1
                self.orders.append(
                    OpenOrder(i.symbol, i.side, i.price, i.qty, 0.0, i.client_id, str(self.seq), self.t)
                )
        return decision

    def step(self, bid=None, **kw):
        if bid is not None:
            self.bids = {s: bid for s in self.symbols}
        snap = self.snapshot(**kw)
        decision, self.state = decide_paired(snap, self.state, self.params, self.risk)
        return self.apply(decision)

    def fill(self, order: OpenOrder):
        self.orders = [o for o in self.orders if o.client_id != order.client_id]
        sign = 1.0 if order.side == "buy" else -1.0
        self.qty[order.symbol] = round(self.qty[order.symbol] + sign * order.qty, 12)
        self.cash -= sign * order.price * order.qty
        self.pending.append(
            {"symbol": order.symbol, "side": order.side, "price": order.price,
             "qty": order.qty, "client_id": order.client_id}
        )

    def match(self, bid):
        """Move the market to ``bid`` and fill every resting order it trades through."""
        self.bids = {s: bid for s in self.symbols}
        for o in list(self.orders):
            if (o.side == "buy" and bid < o.price) or (o.side == "sell" and bid + 0.02 > o.price):
                self.fill(o)

    def buys(self):
        return sorted((o for o in self.orders if o.side == "buy"), key=lambda o: o.price)

    def sells(self):
        return sorted((o for o in self.orders if o.side == "sell"), key=lambda o: o.price)

    def equity(self):
        return self.cash + sum(q * (self.bids[s] + 0.01) for s, q in self.qty.items())

    def exposure(self):
        held = sum(q * (self.bids[s] + 0.01) for s, q in self.qty.items())
        return held + sum(o.price * o.remaining for o in self.orders if o.side == "buy")


def level(sim, i, side="buy", symbol=SYM):
    gs = sim.state[symbol]
    return flt.round_price(side, gs.levels[i], FILT.tick_size)


# ----------------------------------------------------------------------------- variant B
def test_b_initial_buys_sit_on_levels_below_bid():
    sim = Sim()
    sim.step()
    gs = sim.state[SYM]
    assert len(gs.levels) == PARAMS.levels + 1
    assert gs.anchor == pytest.approx(100.01)
    # geometric range anchor * (1 +- step) ** (levels / 2) around the middle line
    assert gs.levels[-1] == pytest.approx(gs.anchor * 1.02**5)
    assert gs.levels[0] == pytest.approx(gs.anchor * 1.02**-5)
    assert [o.price for o in sim.buys()] == [level(sim, i) for i in range(5)]
    assert not sim.sells()
    # equal quote per level (within the qty step)
    quotes = [o.price * o.qty for o in sim.buys()]
    assert max(quotes) - min(quotes) < 0.2
    assert sum(quotes) <= RISK.per_symbol_cap * CASH + 1e-6


def test_b_buy_fill_places_sell_one_level_up_and_sell_fill_rearms_buy():
    sim = Sim()
    sim.step()
    buy4 = sim.buys()[-1]
    assert buy4.price == level(sim, 4)
    sim.bids = {SYM: 98.0}
    sim.fill(buy4)
    sim.step(bid=98.0)
    assert sim.state[SYM].held == frozenset({4})
    (sell,) = sim.sells()
    assert sell.price == level(sim, 5, "sell")
    assert sell.qty == pytest.approx(buy4.qty)
    # the filled level does not re-buy while held
    assert level(sim, 4) not in [o.price for o in sim.buys()]
    sim.fill(sell)
    sim.step(bid=99.0)
    assert sim.state[SYM].held == frozenset()
    assert not sim.sells()
    assert level(sim, 4) in [o.price for o in sim.buys()]  # re-armed at level 4


def test_b_two_fills_in_one_cycle_each_get_their_own_sell():
    sim = Sim()
    sim.step()
    b4, b3 = sim.buys()[-1], sim.buys()[-2]
    sim.fill(b4)
    sim.fill(b3)
    sim.step(bid=96.0)
    assert sim.state[SYM].held == frozenset({3, 4})
    assert [(o.price, o.qty) for o in sim.sells()] == [
        (level(sim, 4, "sell"), b3.qty),
        (level(sim, 5, "sell"), b4.qty),
    ]


@pytest.mark.parametrize("bid", [100.0, 97.5, 95.0, 91.0, 120.0])
def test_b_no_buy_at_or_above_the_bid(bid):
    sim = Sim()
    sim.step()  # fixes the anchor at 100.01
    sim.step(bid=bid)
    buys = sim.buys()
    assert all(o.price < bid for o in buys)
    expect = [
        i for i in range(PARAMS.levels) if flt.round_price("buy", sim.state[SYM].levels[i], 0.01) < bid
    ]
    assert len(buys) == len(expect)


def test_b_recentres_only_when_flat_and_after_24h_outside_range():
    sim = Sim()
    sim.step()
    anchor0 = sim.state[SYM].anchor
    top = sim.state[SYM].levels[-1]
    sim.step(bid=130.0)  # far above the range
    assert 130.0 > top
    # 23h45m outside: still the old grid (timer started at the first outside observation)
    for _ in range(94):
        sim.step(bid=130.0)
    assert sim.state[SYM].anchor == anchor0
    # back inside resets the timer ...
    sim.step(bid=100.0)
    assert sim.state[SYM].out_of_range_since is None
    for _ in range(96):  # a fresh timer: 96 outside observations span 23h45m
        sim.step(bid=130.0)
    assert sim.state[SYM].anchor == anchor0
    # ... and 24h of continuous time outside, while flat, re-centres (cancelling the old orders)
    sim.step(bid=130.0)
    assert sim.state[SYM].anchor == pytest.approx(130.01)
    assert not sim.orders  # the re-centre cycle only cancels
    sim.step(bid=130.0)
    assert sim.buys() and all(o.price < 130.0 for o in sim.buys())
    assert sim.state[SYM].levels[0] == pytest.approx(130.01 * 1.02**-5)


def test_b_does_not_recentre_while_holding():
    sim = Sim()
    sim.step()
    sim.fill(sim.buys()[-1])
    sim.step(bid=98.0)
    anchor0 = sim.state[SYM].anchor
    for _ in range(4 * 24 * 3):  # three days far above the range (the sell never fills here)
        sim.bids = {SYM: 140.0}
        sim.step(bid=140.0)
    assert sim.state[SYM].anchor == anchor0
    assert sim.state[SYM].held == frozenset({4})


def test_b_sell_is_never_inside_the_spread():
    sim = Sim()
    sim.step()
    b4 = sim.buys()[-1]
    sim.fill(b4)
    sim.step(bid=105.0)  # price already ran past the level-5 line (100.01)
    (sell,) = sim.sells()
    assert sell.price >= 105.0 + 0.02 + 0.01 - 1e-9
    assert sell.qty == pytest.approx(b4.qty)


def test_b_regime_gate_blocks_new_buys_but_keeps_sells():
    sim = Sim()
    sim.step()
    sim.fill(sim.buys()[-1])
    sim.step(bid=98.0)
    (sell,) = sim.sells()
    assert sim.buys()
    d = sim.step(bid=98.0, features={SYM: _feat(regime_ok=False)})
    assert not any(isinstance(i, PlaceOrder) and i.side == "buy" for i in d.intents)
    assert not sim.buys()  # resting buys are withdrawn
    assert [o.client_id for o in sim.sells()] == [sell.client_id]  # the sell was not touched
    assert not any(isinstance(i, CancelOrder) and i.client_id == sell.client_id for i in d.intents)
    # a freshly filled level still gets its sell while the gate is shut
    sim.step(bid=98.0)  # gate open again: buys back
    b3 = sim.buys()[-1]
    sim.fill(b3)
    sim.step(bid=96.0, features={SYM: _feat(regime_ok=False)})
    assert len(sim.sells()) == 2 and not sim.buys()


def test_b_total_exposure_stays_within_cap_on_a_random_path():
    syms = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    sim = Sim(symbols=syms)
    rng = random.Random(7)
    px = 100.0
    peak_expo = 0.0
    for _ in range(600):
        px *= 1.0 + rng.gauss(0, 0.006)
        sim.match(px)
        sim.peak = max(sim.peak, sim.equity())
        sim.step(bid=px)
        eq = sim.equity()
        assert sim.exposure() <= RISK.total_exposure_cap * eq + 1e-6
        for s in syms:
            per = sim.qty[s] * (px + 0.01) + sum(
                o.price * o.remaining for o in sim.orders if o.symbol == s and o.side == "buy"
            )
            assert per <= RISK.per_symbol_cap * eq + 1e-6
        assert sim.cash >= 0
        peak_expo = max(peak_expo, sim.exposure() / eq)
    assert peak_expo > 0.15  # not vacuous: the grid really deployed capital


def test_b_tight_cap_is_respected():
    risk = RiskParams(
        total_exposure_cap=0.05, per_symbol_cap=0.05, portfolio_halt_dd=0.15, symbol_halt_dd=0.25
    )
    sim = Sim(risk=risk)
    sim.step()
    assert 0 < sum(o.price * o.qty for o in sim.buys()) <= 0.05 * CASH + 1e-6


def test_b_stale_snapshot_no_intents_but_fills_are_kept():
    sim = Sim()
    sim.step()
    b4 = sim.buys()[-1]
    sim.fill(b4)
    d = sim.step(bid=98.0, stale=(SYM,))
    assert d.intents == ()
    assert sim.state[SYM].held == frozenset({4})  # the fill was not lost
    sim.step(bid=98.0)
    assert [o.qty for o in sim.sells()] == [b4.qty]


def test_b_halt_cancels_and_flattens():
    sim = Sim()
    sim.step()
    sim.fill(sim.buys()[-1])
    snap = sim.snapshot(peak=CASH * 2)  # equity far below peak -> portfolio halt
    d, _ = decide_paired(snap, sim.state, PARAMS, RISK)
    assert d.halt.active
    cancels = {i.client_id for i in d.intents if isinstance(i, CancelOrder)}
    assert cancels == {o.client_id for o in sim.orders}
    sells = [i for i in d.intents if isinstance(i, PlaceOrder)]
    assert len(sells) == 1 and sells[0].side == "sell" and not sells[0].post_only


def test_b_cold_symbol_cancels_buys_and_places_nothing():
    sim = Sim()
    sim.step()
    assert sim.buys()
    sim.step(features={SYM: _feat(warm=False)})
    assert not sim.orders


# ----------------------------------------------------------------------------- variant C
CRISK = RiskParams(
    total_exposure_cap=0.3, per_symbol_cap=0.2, portfolio_halt_dd=0.15, symbol_halt_dd=0.25
)
CSYMS = ("AAAUSDT", "BBBUSDT")
BAND = RebalanceParams(band=0.1)
TARGET = 0.3 * CASH / 2  # two equal-vol coins -> 1500 each


def csnap(qty=None, orders=(), equity=CASH, bid=100.0, features=None, stale=(), peak=None):
    qty = qty or {}
    markets = {s: SymbolMarket(s, T0, bid, bid + 0.02, bid, FILT) for s in CSYMS}
    held = {s: Position(s, q, 100.0, 0.0) for s, q in qty.items() if q > 0}
    held_value = sum(q * (bid + 0.01) for q in qty.values())
    reserved = sum(o.price * o.remaining for o in orders if o.side == "buy")
    pf = PortfolioState(
        ts=T0, cash_free=equity - held_value - reserved, cash_reserved=reserved,
        positions=held, open_orders=tuple(orders), peak_equity=peak if peak else equity,
        halt=HaltState(),
    )
    feats = features if features is not None else {s: _feat() for s in CSYMS}
    return Snapshot(ts=T0, markets=markets, bars={}, portfolio=pf, stale=stale, features=feats)


def _oo(sym, side, price, qty, cid):
    return OpenOrder(sym, side, price, qty, 0.0, cid, "1", T0)


def test_c_inside_band_places_nothing():
    d = decide_rebalance(csnap(qty={s: 15.0 for s in CSYMS}), BAND, CRISK)  # value ~1500
    assert d.intents == ()
    d = decide_rebalance(csnap(qty={s: 16.0 for s in CSYMS}), BAND, CRISK)  # +6.7% < band
    assert d.intents == ()


def test_c_below_target_buys_the_difference_at_bid_minus_one_tick():
    d = decide_rebalance(csnap(qty={SYM_: 10.0 for SYM_ in CSYMS}), BAND, CRISK)  # ~1000 vs 1500
    assert len(d.intents) == 2
    for i in d.intents:
        assert isinstance(i, PlaceOrder) and i.side == "buy" and i.post_only
        assert i.price == pytest.approx(99.99)
        gap = TARGET - 10.0 * 100.01
        assert i.qty == pytest.approx(gap / 99.99, abs=0.0011)
        assert i.qty * i.price <= gap + 1e-6  # rounded down, never overshoots


def test_c_above_target_sells_the_difference_at_ask_plus_one_tick():
    d = decide_rebalance(csnap(qty={CSYMS[0]: 20.0, CSYMS[1]: 15.0}), BAND, CRISK)
    (o,) = d.intents
    assert o.symbol == CSYMS[0] and o.side == "sell" and o.post_only
    assert o.price == pytest.approx(100.03)
    assert o.qty == pytest.approx((20.0 * 100.01 - TARGET) / 100.03, abs=0.0011)


def test_c_sell_never_exceeds_the_free_position():
    snap = csnap(qty={CSYMS[0]: 20.0, CSYMS[1]: 15.0})
    d = decide_rebalance(snap, RebalanceParams(band=0.05), CRISK)
    for i in d.intents:
        assert i.side == "sell" and i.qty <= 20.0


@pytest.mark.parametrize("held", [0.0, 3.0, 9.0, 12.0, 14.0, 17.0, 19.0, 25.0, 40.0])
@pytest.mark.parametrize("bid", [100.0, 97.3])
def test_c_post_only_prices_are_never_marketable(held, bid):
    snap = csnap(qty={s: held for s in CSYMS}, bid=bid)
    d = decide_rebalance(snap, BAND, CRISK)
    m = snap.markets[CSYMS[0]]
    for i in d.intents:
        assert isinstance(i, PlaceOrder) and i.post_only
        if i.side == "buy":
            assert i.price < m.bid
        else:
            assert i.price > m.ask


def test_c_at_most_one_resting_order_per_coin():
    a = CSYMS[0]
    orders = [
        _oo(a, "buy", 99.9, 3.0, "vv-1"),
        _oo(a, "buy", 99.5, 2.0, "vv-2"),
        _oo(a, "sell", 101.5, 1.0, "vv-3"),
    ]
    snap = csnap(qty={s: 10.0 for s in CSYMS}, orders=orders)
    d = decide_rebalance(snap, BAND, CRISK)
    resting = {o.client_id for o in orders}
    for i in d.intents:
        if isinstance(i, CancelOrder):
            resting.discard(i.client_id)
    placed_a = [i for i in d.intents if isinstance(i, PlaceOrder) and i.symbol == a]
    assert len(resting) + len(placed_a) <= 1
    # inside the band every resting order goes
    snap = csnap(qty={s: 15.0 for s in CSYMS}, orders=orders)
    d = decide_rebalance(snap, BAND, CRISK)
    assert {i.client_id for i in d.intents if isinstance(i, CancelOrder)} == {"vv-1", "vv-2", "vv-3"}
    assert not any(isinstance(i, PlaceOrder) for i in d.intents)


def test_c_keeps_resting_order_until_target_moves_more_than_half_band():
    a, b = CSYMS
    # buy 5.0 @ 99.99 on top of 10 held -> implied value ~1499.85 == target: keep as is
    orders = [_oo(a, "buy", 99.99, 5.0, "vv-keep")]
    d = decide_rebalance(csnap(qty={a: 10.0, b: 15.0}, orders=orders), BAND, CRISK)
    assert d.intents == ()
    # the wallet grows 3% (target +3% < band/2 = 5%): still kept
    d = decide_rebalance(csnap(qty={a: 10.0, b: 15.0}, orders=orders, equity=CASH * 1.03), BAND, CRISK)
    assert d.intents == ()
    # +10% target (> band / 2): replaced by a new order at the new size
    d = decide_rebalance(csnap(qty={a: 10.0, b: 15.0}, orders=orders, equity=CASH * 1.10), BAND, CRISK)
    assert CancelOrder(a, "vv-keep") in d.intents
    (new,) = [i for i in d.intents if isinstance(i, PlaceOrder)]
    assert new.symbol == a and new.side == "buy" and new.qty > 5.0


def test_c_wrong_side_resting_order_is_replaced():
    a, b = CSYMS
    orders = [_oo(a, "buy", 99.99, 5.0, "vv-old")]
    # the holding rose well above target: the buy must go and a sell takes over
    d = decide_rebalance(csnap(qty={a: 20.0, b: 15.0}, orders=orders), BAND, CRISK)
    assert CancelOrder(a, "vv-old") in d.intents
    assert [i.side for i in d.intents if isinstance(i, PlaceOrder)] == ["sell"]


def test_c_ignores_regime_but_needs_warm_features():
    feats = {s: _feat(regime_ok=False) for s in CSYMS}
    d = decide_rebalance(csnap(qty={s: 10.0 for s in CSYMS}, features=feats), BAND, CRISK)
    assert len(d.intents) == 2  # no regime gate
    feats = {s: _feat(warm=False) for s in CSYMS}
    d = decide_rebalance(csnap(qty={s: 10.0 for s in CSYMS}, features=feats), BAND, CRISK)
    assert d.intents == ()


def test_c_inverse_vol_targets_and_exposure_cap():
    a, b = CSYMS
    feats = {a: _feat(sigma=0.01), b: _feat(sigma=0.02)}
    d = decide_rebalance(csnap(features=feats), RebalanceParams(band=0.1), CRISK)
    by_sym = {i.symbol: i.price * i.qty for i in d.intents}
    assert by_sym[a] == pytest.approx(2 * by_sym[b], rel=0.01)  # calmer coin gets 2x
    assert sum(by_sym.values()) <= CRISK.total_exposure_cap * CASH + 1e-6


def test_c_stale_none_and_halt_flattens():
    assert decide_rebalance(csnap(qty={s: 10.0 for s in CSYMS}, stale=(CSYMS[0],)), BAND, CRISK).intents == ()
    snap = csnap(qty={CSYMS[0]: 10.0}, peak=CASH * 2)
    d = decide_rebalance(snap, BAND, CRISK)
    assert d.halt.active
    sells = [i for i in d.intents if isinstance(i, PlaceOrder)]
    assert len(sells) == 1 and sells[0].side == "sell" and not sells[0].post_only
