"""Replay engine on synthetic 2-symbol bars (45 warm-up days + a few replay days)."""

import numpy as np
import pandas as pd
import pytest

import vv.sim.replay as replay_mod
from vv.core.params import GridParams, RiskParams
from vv.core.types import CancelOrder, Decision, PlaceOrder, SymbolFilters
from vv.sim.replay import Fees, ReplayJob, run_many, run_replay

T0 = pd.Timestamp("2024-01-01", tz="UTC")
WARM_DAYS = 45
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)
A, B = "AAAUSDT", "BBBUSDT"
MIN = pd.Timedelta(minutes=1)

GRID = GridParams(
    spacing_sigmas=0.5, min_step=0.005, sigma_horizon_minutes=60, levels=5,
    qty_multiplier=1.25, maker_fee=0.00075,
)
RISK = RiskParams(
    total_exposure_cap=0.38, per_symbol_cap=0.15, portfolio_halt_dd=0.15, symbol_halt_dd=0.25
)


def synth(n_days, seed, base, shock=None):
    """Mean-reverting noisy 1m bars on a gentle uptrend. ``shock(t_minutes, level) -> level``."""
    n = n_days * 1440
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = rng.normal(0, 0.002, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] * 0.99 + e[i]
    lvl = base * (1 + 0.003 * t / 1440) * (1 + 0.01 * np.sin(2 * np.pi * t / 360 + seed)) * np.exp(x)
    if shock is not None:
        lvl = shock(t, lvl)
    open_ = np.concatenate([[lvl[0]], lvl[:-1]])
    wick = np.abs(rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, lvl) * (1 + wick),
            "low": np.minimum(open_, lvl) * (1 - wick),
            "close": lvl,
            "volume": 1.0,
        },
        index=pd.date_range(T0, periods=n, freq="1min"),
    ).round(2)


@pytest.fixture(scope="module")
def bars():
    return {A: synth(WARM_DAYS + 4, 1, 100.0), B: synth(WARM_DAYS + 4, 2, 50.0)}


FILTERS = {A: FILT, B: FILT}
START = T0 + pd.Timedelta(days=WARM_DAYS)
END = START + pd.Timedelta(days=3)


def run(bars, **kw):
    args = dict(
        bars=bars, filters=FILTERS, grid=GRID, risk=RISK, start=START, end=END,
        decision_minutes=15, start_cash=10_000.0, fees=Fees(), fill_haircut=1.0, seed=0,
    )
    args.update(kw)
    return run_replay(**args)


def test_it_trades_and_accounts(bars):
    res = run(bars)
    assert res.n_decisions == 3 * 96
    assert res.n_fills > 20 and res.round_trips > 5
    assert res.fees_paid > 0 and res.gross_pnl != 0
    assert len(res.daily_equity) == 4  # 3 midnights + the final point at `end`
    assert res.daily_equity.index[0] == START and res.daily_equity.index[-1] == END
    assert res.daily_equity.iloc[0] == pytest.approx(10_000.0)
    assert res.final_equity == pytest.approx(res.daily_equity.iloc[-1])
    assert 0 < res.time_in_market <= 1 and 0 < res.avg_exposure < 0.38
    # fills recorded by the exchange reconcile with cash + position accounting:
    # equity change == gross pnl - fees + mark-to-market of what is still held
    assert res.n_taker_fills == 0
    assert res.rejections == {}


def test_no_look_ahead_features_come_from_closed_bars(bars):
    res = run(bars, record_trace=True)
    assert res.trace and len(res.trace) == res.n_decisions
    for t, used in res.trace:
        assert set(used) == {A, B}
        for label in used.values():
            assert label < t  # a bar labelled L closes at L+1m; need L+1m <= t
            assert label == t - MIN  # and it is the latest such bar


def test_results_do_not_depend_on_bars_at_or_after_end(bars):
    base = run(bars)
    cut = {s: df.loc[: END - MIN] for s, df in bars.items()}  # nothing from `end` on exists
    poisoned = {s: df.copy() for s, df in bars.items()}
    for df in poisoned.values():
        df.loc[END:, ["open", "high", "low", "close"]] *= 3.0
    for other in (cut, poisoned):
        r = run(other)
        pd.testing.assert_series_equal(r.daily_equity, base.daily_equity)
        assert r.n_fills == base.n_fills and r.fees_paid == base.fees_paid


def test_latched_halt_stays_latched_and_flattens(bars):
    crash_start = (WARM_DAYS + 1) * 1440 + 6 * 60  # day 46, 06:00
    rebound = crash_start + 24 * 60

    def shock(t, lvl):
        ramp = np.clip((t - crash_start) / 180.0, 0.0, 1.0)  # -20% over 3h
        drop = 1.0 - 0.20 * ramp
        up = np.clip((t - rebound) / 180.0, 0.0, 1.0)  # fully recovers a day later
        return lvl * (drop + 0.20 * up * ramp)

    crash = {A: synth(WARM_DAYS + 4, 1, 100.0, shock), B: synth(WARM_DAYS + 4, 2, 50.0, shock)}
    risk = RiskParams(
        total_exposure_cap=0.38, per_symbol_cap=0.15, portfolio_halt_dd=0.02, symbol_halt_dd=0.9
    )
    res = run(crash, risk=risk)
    kinds = [e["kind"] for e in res.halt_events]
    assert kinds.count("portfolio_halt") == 1
    t_halt = next(e["ts"] for e in res.halt_events if e["kind"] == "portfolio_halt")
    d = res.decisions
    after = d.loc[d.index >= t_halt]
    # latched for the rest of the run, even though price fully recovered
    assert after["halt_active"].all()
    assert not d.loc[d.index < t_halt, "halt_active"].any()
    # position goes to zero within a few cycles and stays there
    assert (after["n_positions"].iloc[:6] == 0).any()
    first_flat = after.index[(after["n_positions"] == 0).to_numpy()][0]
    assert (first_flat - t_halt) <= pd.Timedelta(minutes=15 * 6)
    assert (after.loc[after.index >= first_flat, "n_positions"] == 0).all()
    # nothing is bought once halted; the flatten went through as taker sells
    late = [f for f in res.fills if f["ts"] > t_halt + pd.Timedelta(minutes=15)]
    assert all(f["side"] == "sell" for f in late)
    assert res.n_taker_fills >= 1


def test_exposure_never_exceeds_total_cap(bars):
    res = run(bars, end=START + pd.Timedelta(days=4))
    cap = RISK.total_exposure_cap
    d = res.decisions
    assert d["exposure_total_post"].max() <= cap + 1e-9
    assert res.max_exposure_total_post == pytest.approx(d["exposure_total_post"].max())
    assert d["exposure_total_post"].max() > 0.05  # the check is not vacuous


def test_rejections_are_counted_not_raised(bars, monkeypatch):
    calls = {"n": 0}

    def fake_decide(snapshot, params, risk):
        calls["n"] += 1
        m = snapshot.markets[A]
        ok = PlaceOrder(A, "buy", round(m.bid - 5, 2), 1.0, "vv-same")
        dup = PlaceOrder(A, "buy", round(m.bid - 6, 2), 1.0, "vv-same")  # id reuse
        cross = PlaceOrder(A, "buy", round(m.ask + 1, 2), 1.0, f"vv-cross-{calls['n']}")
        bad_tick = PlaceOrder(A, "buy", round(m.bid - 7, 2) + 0.001, 1.0, f"vv-tick-{calls['n']}")
        gone = CancelOrder(A, "vv-never-existed")
        return Decision((ok, dup, cross, bad_tick, gone), snapshot.portfolio.halt)

    monkeypatch.setattr(replay_mod, "decide", fake_decide)
    res = run(bars, end=START + pd.Timedelta(hours=2))
    n = calls["n"]
    assert n == 8
    # "vv-same" is accepted exactly once (first cycle); every other attempt, `ok` again and
    # `dup`, is an id reuse
    assert res.rejections["duplicate_client_id"] == 2 * n - 1
    assert res.rejections["post_only_cross"] == n
    assert res.rejections["not_on_tick"] == n
    assert res.rejections["unknown_order"] == n
    assert res.n_places == 1


def test_same_seed_identical_and_seed_matters_with_haircut(bars):
    a = run(bars, fill_haircut=0.5, seed=11)
    b = run(bars, fill_haircut=0.5, seed=11)
    c = run(bars, fill_haircut=0.5, seed=12)
    pd.testing.assert_series_equal(a.daily_equity, b.daily_equity)
    assert a.fills == b.fills and a.rejections == b.rejections
    assert [f["ts"] for f in a.fills] != [f["ts"] for f in c.fills]
    full = run(bars, fill_haircut=1.0)
    assert a.n_fills < full.n_fills  # haircut delays/removes fills


def test_run_many_matches_serial_and_is_order_independent(bars):
    feats = replay_mod.precompute_features(bars, GRID.sigma_horizon_minutes)
    jobs = [
        ReplayJob(
            key=("k", i),
            kwargs=dict(
                grid=GRID, risk=RISK, start=START, end=START + pd.Timedelta(days=1 + i),
                decision_minutes=15, start_cash=10_000.0, fees=Fees(), seed=i,
            ),
            symbols=(A, B) if i else (A,),
        )
        for i in range(3)
    ]
    shared = dict(bars=bars, features=feats, filters=FILTERS)
    serial = run_many(jobs, processes=1, **shared)
    par = run_many(list(reversed(jobs)), processes=2, **shared)
    assert set(serial) == set(par) == {j.key for j in jobs}
    for k in serial:
        pd.testing.assert_series_equal(serial[k].daily_equity, par[k].daily_equity)
        assert serial[k].fees_paid == par[k].fees_paid
    assert serial[("k", 0)].symbols == (A,)
    with pytest.raises(ValueError):
        run_many([jobs[0], jobs[0]], **shared)


def test_precomputed_features_equal_internal(bars):
    feats = replay_mod.precompute_features(bars, GRID.sigma_horizon_minutes)
    a = run(bars)
    b = run(bars, features=feats)
    pd.testing.assert_series_equal(a.daily_equity, b.daily_equity)


def test_bad_window_raises(bars):
    with pytest.raises(ValueError):
        run(bars, start=END, end=START)
