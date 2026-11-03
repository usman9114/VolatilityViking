"""WP-S1.7 diagnostics: min_pierce, top-N-day share, exposure audit, cash yield, VR regression."""

import asyncio
import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from research import g1_diagnostics as gd
from vv.core.params import GridParams, RiskParams
from vv.core.types import CancelOrder, PlaceOrder, SymbolFilters
from vv.sim import metrics as m
from vv.sim.fake_exchange import FakeExchange, OrderRejected
from vv.sim.replay import Fees, run_replay

T0 = pd.Timestamp("2026-01-01", tz="UTC")
MIN = pd.Timedelta(minutes=1)
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)
A, B = "AAA/USDT", "BBB/USDT"


def _rows(rows):
    idx = pd.date_range(T0, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


def _make(rows_by_sym, **kw):
    kw.setdefault("start_cash", 10_000.0)
    kw.setdefault("maker_fee", 0.001)
    kw.setdefault("taker_fee", 0.001)
    kw.setdefault("taker_slippage_bps", 0)
    data = {s: _rows(r) for s, r in rows_by_sym.items()}
    return FakeExchange(data, {s: FILT for s in data}, **kw)


def _sync(coro):
    return asyncio.run(coro)


def _buy(cid, price, sym=A, qty=1.0):
    return PlaceOrder(sym, "buy", price, qty, cid, True)


def _sell(cid, price, sym=A, qty=1.0):
    return PlaceOrder(sym, "sell", price, qty, cid, True)


# ------------------------------------------------------------------ min_pierce
def test_min_pierce_zero_is_one_tick_and_positive_needs_proportional_depth():
    # price 99.00, tick 0.01: min_pierce 0.001 -> threshold 99*(1-0.001) - 0.01 = 98.891
    shallow = [(100, 100, 100, 100), (100, 100, 98.98, 99.5), (99.5, 99.5, 99.5, 99.5)]
    deep = [(100, 100, 100, 100), (100, 100, 98.88, 99.5), (99.5, 99.5, 99.5, 99.5)]

    def filled(rows, **kw):
        ex = _make({A: rows}, **kw)
        ex.advance_to(T0 + MIN)
        _sync(ex.create_order(_buy("b", 99.00)))
        ex.advance_to(T0 + 2 * MIN)
        return len(ex.fills) == 1

    assert filled(shallow)
    assert filled(shallow, min_pierce=0.0)
    assert not filled(shallow, min_pierce=0.001)  # pierces by ~1 tick only: blocked
    assert filled(deep, min_pierce=0.001)  # 98.88 < 98.891: fills
    assert not filled(deep, min_pierce=0.002)  # threshold 99*0.998 - 0.01 = 98.788


def test_min_pierce_sells_are_symmetric():
    # sell at 101.00 needs high > 101*(1+mp) + tick
    base = [(100, 100, 100, 100), (100, 100, 100, 100)]
    shallow = base + [(100, 101.02, 100, 101), (101, 101, 101, 101)]
    deep = base + [(100, 101.12, 100, 101), (101, 101, 101, 101)]

    def filled(rows, **kw):
        ex = _make({A: rows}, **kw)
        ex.advance_to(T0 + MIN)
        _sync(ex.create_order(PlaceOrder(A, "buy", 100.0, 1.0, "b", False)))
        ex.advance_to(T0 + 2 * MIN)  # taker buy fills at the open of bar 1
        _sync(ex.create_order(_sell("s", 101.00)))
        ex.advance_to(T0 + 4 * MIN)
        return any(f["client_id"] == "s" for f in ex.fills)

    assert filled(shallow)
    assert filled(shallow, min_pierce=0.0)
    assert not filled(shallow, min_pierce=0.001)  # threshold 101*1.001 + 0.01 = 101.111
    assert filled(deep, min_pierce=0.001)  # 101.12 > 101.111


def test_min_pierce_does_not_touch_taker_fills_and_is_validated():
    rows = [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)]
    ex = _make({A: rows}, min_pierce=0.01)
    ex.advance_to(T0 + MIN)
    _sync(ex.create_order(PlaceOrder(A, "buy", 101.0, 1.0, "t", False)))
    ex.advance_to(T0 + 2 * MIN)
    assert [f["liquidity"] for f in ex.fills] == ["taker"]
    for bad in (-0.001, 1.0, 2.0):
        with pytest.raises(ValueError):
            _make({A: rows}, min_pierce=bad)


def _golden_scenario(**ex_kw):
    """Two symbols, deterministic path, many post-only orders at varied depths; digest of fills."""
    n = 600
    t = np.arange(n)
    path = {
        A: 100 + 4 * np.sin(t / 7.0) + 2.5 * np.sin(t / 2.9),
        B: 50 + 2 * np.sin(t / 5.0) + 1.5 * np.sin(t / 3.3),
    }
    data = {}
    for s, lvl in path.items():
        close = np.round(lvl, 2)
        open_ = np.concatenate([[close[0]], close[:-1]])
        wig = 0.02 * (1 + (t % 5))
        high = np.round(np.maximum(open_, close) + wig, 2)
        low = np.round(np.minimum(open_, close) - wig, 2)
        data[s] = _rows(list(zip(open_, high, low, close, strict=True)))
    ex = FakeExchange(
        data, {s: FILT for s in data}, 10_000.0, maker_fee=0.00075, taker_fee=0.001,
        taker_slippage_bps=10, fill_haircut=0.8, seed=3, **ex_kw,
    )
    n_orders = 0
    for k in range(1, n - 1):
        ex.advance_to(T0 + k * MIN)
        for s, df in data.items():
            last = float(df["close"].iloc[k - 1])
            d = 1 + (k + (s == B)) % 6
            if k % 2 == 0:
                px = math.floor((last - d * 0.07) * 100) / 100
                od = PlaceOrder(s, "buy", px, 1.0, f"b{k}{s}", True)
            else:
                px = math.ceil((last + d * 0.06) * 100) / 100
                od = PlaceOrder(s, "sell", px, 1.0, f"s{k}{s}", True)
            try:
                _sync(ex.create_order(od))
                n_orders += 1
            except OrderRejected:
                pass
        for o in list(_sync(ex.fetch_open_orders())):
            if (o.ts - T0) / MIN <= k - 8:
                _sync(ex.cancel_order(CancelOrder(o.symbol, o.client_id)))
    h = hashlib.sha256()
    for f in ex.fills:
        h.update(
            f"{f['ts'].isoformat()}|{f['symbol']}|{f['side']}|{f['price']:.8f}|{f['qty']:.8f}|"
            f"{f['client_id']}|{f['liquidity']}\n".encode()
        )
    h.update(f"{ex.cash_free:.6f}".encode())
    return len(ex.fills), h.hexdigest()


# Digest recorded from the code *before* min_pierce existed (same scenario, no parameter).
GOLDEN_FILLS = 712
GOLDEN_DIGEST = "7044c488f8b7cfc86c0541a7dddee34de3fa1fabc216d8ca8af2b8e4fd7509a2"


def test_min_pierce_default_is_byte_identical_to_pre_change_behaviour():
    n, digest = _golden_scenario()
    assert n > 50, "scenario must actually trade"
    assert (n, digest) == (GOLDEN_FILLS, GOLDEN_DIGEST)
    n0, digest0 = _golden_scenario(min_pierce=0.0)
    assert (n0, digest0) == (GOLDEN_FILLS, GOLDEN_DIGEST)


def test_min_pierce_positive_changes_the_scenario():
    n, digest = _golden_scenario(min_pierce=0.0015)
    assert n < GOLDEN_FILLS and digest != GOLDEN_DIGEST


# ------------------------------------------------- replay-level pierce wrapper
GRID = GridParams(
    spacing_sigmas=0.5, min_step=0.005, sigma_horizon_minutes=60, levels=5,
    qty_multiplier=1.25, maker_fee=0.00075,
)
RISK = RiskParams(
    total_exposure_cap=0.38, per_symbol_cap=0.15, portfolio_halt_dd=0.15, symbol_halt_dd=0.25
)


def _synth_bars(n_days, seed, base):
    n = n_days * 1440
    rng = np.random.default_rng(seed)
    tt = np.arange(n)
    x = np.zeros(n)
    e = rng.normal(0, 0.002, n)
    for i in range(1, n):
        x[i] = x[i - 1] * 0.99 + e[i]
    lvl = base * (1 + 0.01 * np.sin(2 * np.pi * tt / 360 + seed)) * np.exp(x)
    open_ = np.concatenate([[lvl[0]], lvl[:-1]])
    wick = np.abs(rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, lvl) * (1 + wick),
         "low": np.minimum(open_, lvl) * (1 - wick), "close": lvl, "volume": 1.0},
        index=pd.date_range(T0, periods=n, freq="1min"),
    ).round(2)


def test_pierce_wrapper_threads_min_pierce_into_run_replay_and_restores():
    import vv.sim.replay as replay_mod

    bars = {"AAAUSDT": _synth_bars(48, 1, 100.0)}
    flt = {"AAAUSDT": FILT}
    start, end = T0 + pd.Timedelta(days=45), T0 + pd.Timedelta(days=47)
    kw = dict(bars=bars, filters=flt, grid=GRID, risk=RISK, start=start, end=end,
              start_cash=10_000.0, fees=Fees())
    orig = replay_mod.FakeExchange
    base = run_replay(**kw)
    with gd.pierce_exchange(0.0):
        same = run_replay(**kw)
    with gd.pierce_exchange(0.0015):
        deep = run_replay(**kw)
    assert replay_mod.FakeExchange is orig
    assert base.n_fills > 10
    assert [f["client_id"] for f in same.fills] == [f["client_id"] for f in base.fills]
    assert same.final_equity == base.final_equity
    assert deep.n_fills < base.n_fills


# ------------------------------------------------------------- top-N-day share
def test_top_n_day_share_on_known_series():
    pnl = np.array([10.0, -4.0, 6.0, 1.0, 3.0, -2.0, 0.0, 4.0])  # total 18
    assert m.top_n_day_pnl_share(pnl, 1) == pytest.approx(10 / 18)
    assert m.top_n_day_pnl_share(pnl, 2) == pytest.approx(16 / 18)
    assert m.top_n_day_pnl_share(pnl, 3) == pytest.approx(20 / 18)  # >100%: losing days offset
    assert m.top_n_day_pnl_share(pnl, 100) == pytest.approx(1.0)  # all days
    # a flat or losing book has no meaningful share
    assert math.isnan(m.top_n_day_pnl_share(np.array([1.0, -1.0]), 1))
    assert math.isnan(m.top_n_day_pnl_share(np.array([-3.0, 1.0]), 1))
    assert math.isnan(m.top_n_day_pnl_share(np.array([]), 10))


def test_daily_pnl_is_the_difference_of_the_equity_curve():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    eq = pd.Series([100.0, 110.0, 105.0, 111.0], index=idx)
    assert list(m.daily_pnl(eq)) == pytest.approx([10.0, -5.0, 6.0])
    # 10 + 6 = 16 of the total 11 -> 145%
    assert m.top_n_day_pnl_share(m.daily_pnl(eq), 2) == pytest.approx(16 / 11)


# ------------------------------------------------------------ exposure audit
def test_exposure_audit_on_hand_built_trace():
    # minute samples; each sample holds until the next one (time-weighted)
    idx = pd.DatetimeIndex(
        [T0, T0 + 1 * MIN, T0 + 3 * MIN, T0 + 4 * MIN], name="ts"
    )
    equity = pd.Series([1000.0, 1000.0, 1100.0, 1100.0], index=idx)
    held = pd.Series([0.0, 100.0, 200.0, 50.0], index=idx)  # USDT at mark
    resting = pd.Series([300.0, 200.0, 100.0, 100.0], index=idx)  # resting buy notional
    out = m.exposure_audit(held, resting, equity)
    # weights 1, 2, 1 minutes (last sample carries no duration)
    # held/equity: 0, 0.1, 0.181818...; weighted: (0*1 + 0.1*2 + (200/1100)*1)/4
    expect_held = (0 * 1 + 0.1 * 2 + (200 / 1100) * 1) / 4
    # (held+resting)/equity: 0.3, 0.3, 300/1100
    expect_total = (0.3 * 1 + 0.3 * 2 + (300 / 1100) * 1) / 4
    assert out["holdings_exposure"] == pytest.approx(expect_held)
    assert out["total_exposure"] == pytest.approx(expect_total)
    assert out["avg_held_usdt"] == pytest.approx((0 * 1 + 100 * 2 + 200 * 1) / 4)  # 100
    assert out["avg_deployed_usdt"] == pytest.approx((300 * 1 + 300 * 2 + 300 * 1) / 4)  # 300
    # pnl = 1100 - 1000 = 100
    assert out["pnl"] == pytest.approx(100.0)
    assert out["return_on_avg_held"] == pytest.approx(100.0 / 100.0)
    assert out["return_on_avg_deployed"] == pytest.approx(100.0 / 300.0)
    # they are fractions (0..1), not percentages
    assert 0 <= out["holdings_exposure"] <= out["total_exposure"] <= 1


def test_exposure_audit_degenerate_inputs():
    idx = pd.DatetimeIndex([T0, T0 + MIN])
    zero = pd.Series([0.0, 0.0], index=idx)
    eq = pd.Series([100.0, 100.0], index=idx)
    out = m.exposure_audit(zero, zero, eq)
    assert out["holdings_exposure"] == 0.0 and out["total_exposure"] == 0.0
    assert math.isnan(out["return_on_avg_deployed"])  # nothing deployed: undefined, not inf
    one = pd.Series([5.0], index=idx[:1])
    assert math.isnan(m.exposure_audit(one, one, one)["holdings_exposure"])


def test_deployed_from_decisions_and_summarize_keep_the_old_field():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    dec = pd.DataFrame(
        {"equity": [1000.0, 1000.0, 1100.0], "exposure": [0.0, 0.01, 0.02],
         "exposure_total_post": [0.1, 0.3, 0.2]}, index=idx,
    )
    d = m.deployed_from_decisions(dec)
    assert d["avg_total_exposure"] == pytest.approx(0.2)
    assert d["avg_deployed_usdt"] == pytest.approx((100 + 300 + 220) / 3)
    eq = pd.Series([1000.0, 1050.0, 1100.0], index=idx)
    s = m.summarize(eq, avg_exposure=0.01, **d)
    assert s["avg_exposure"] == 0.01  # old field kept as is
    assert s["avg_total_exposure"] == pytest.approx(0.2)
    assert s["return_on_avg_deployed"] == pytest.approx(100.0 / ((100 + 300 + 220) / 3))
    assert math.isnan(m.summarize(eq)["return_on_avg_deployed"])  # nothing supplied
    assert math.isnan(m.deployed_from_decisions(dec.iloc[:0])["avg_deployed_usdt"])


# ----------------------------------------------------------------- cash yield
def test_cash_yield_and_excess_over_cash():
    days = 365
    idx = pd.DatetimeIndex([T0, T0 + pd.Timedelta(days=days)])
    assert m.cash_yield_return(idx) == pytest.approx(0.04)
    half = pd.DatetimeIndex([T0, T0 + pd.Timedelta(days=days / 2)])
    assert m.cash_yield_return(half) == pytest.approx(1.04**0.5 - 1)
    curve = m.cash_yield_curve(idx, 1000.0)
    assert list(curve) == pytest.approx([1000.0, 1040.0])
    eq = pd.Series([1000.0, 1100.0], index=idx)
    s = m.summarize(eq)
    assert s["cash_yield_return"] == pytest.approx(0.04)
    assert s["excess_over_cash"] == pytest.approx(0.10 - 0.04)
    assert s["avg_exposure"] != s["avg_exposure"]  # old field untouched (nan default)


# ----------------------------------------------------------------- VR regression
G0_SNIPPET = """\
## Stability: per calendar year (VR, 95% CI)

| Symbol | Year | 1h VR | 1h CI | 4h VR | 4h CI | DFA Hurst |
|---|---|---|---|---|---|---|
| BTCUSDT | 2019 | 0.934 | [0.894, 0.982] | 0.896 | [0.817, 0.986] | 0.491 |
| ETHUSDT | 2020 | 0.902 | [0.852, 0.961] | 0.834 | [0.751, 0.931] | 0.466 |

## Stability: per calendar year, robust (VR, 95% CI, unadjusted z* p)

| Symbol | Year | 1h VR | 1h CI | 1h p | 4h VR | 4h CI | 4h p |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 2019 | 0.849 | [0.821, 0.878] | 0.000 | 0.783 | [0.733, 0.838] | 0.000 |
| ETHUSDT | 2020 | 0.862 | [0.836, 0.890] | 0.000 | 0.805 | [0.755, 0.860] | 0.000 |

## DFA Hurst exponent
"""


def test_parse_yearly_vr_reads_the_right_table():
    assert gd.parse_yearly_vr(G0_SNIPPET) == {("BTCUSDT", 2019): 0.934, ("ETHUSDT", 2020): 0.902}
    assert gd.parse_yearly_vr(G0_SNIPPET, robust=True) == {
        ("BTCUSDT", 2019): 0.849, ("ETHUSDT", 2020): 0.862,
    }


def test_parse_yearly_vr_on_the_real_g0_file():
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "docs" / "research" / "G0_RESULTS.md"
    vr = gd.parse_yearly_vr(p.read_text())
    assert len(vr) == 62 and vr[("DOGEUSDT", 2019)] == 0.547
    rob = gd.parse_yearly_vr(p.read_text(), robust=True)
    assert len(rob) == 62 and rob[("DOGEUSDT", 2019)] == 0.511


def test_ols_hand_computed():
    out = gd.ols([1, 2, 3, 4], [2, 4, 5, 8])
    # Sxx 5, Sxy 9.5 -> slope 1.9, intercept 0; SSE 0.70 on 2 dof; SST 18.75
    assert out["slope"] == pytest.approx(1.9)
    assert out["intercept"] == pytest.approx(0.0, abs=1e-12)
    assert out["t_slope"] == pytest.approx(1.9 / math.sqrt(0.35 / 5))
    assert out["r2"] == pytest.approx(1 - 0.70 / 18.75)
    assert out["n"] == 4
    assert list(out["resid"]) == pytest.approx([0.1, 0.2, -0.7, 0.4])
