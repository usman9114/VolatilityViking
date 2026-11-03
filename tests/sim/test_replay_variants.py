"""Replay with a pluggable strategy: the default is unchanged; B and C run on synthetic bars."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from test_replay import A, B, END, FILTERS, GRID, RISK, START, WARM_DAYS, run, synth

from vv.core.variants.paired_grid import PairedGridParams, PairedGridStrategy
from vv.core.variants.rebalance import RebalanceParams, RebalanceStrategy
from vv.sim.replay import ReplayJob, precompute_features, run_many


@pytest.fixture(scope="module")
def bars():
    return {A: synth(WARM_DAYS + 4, 1, 100.0), B: synth(WARM_DAYS + 4, 2, 50.0)}


def test_default_strategy_matches_golden_values_from_before_the_change(bars):
    # Captured on the code before ``strategy`` / ``fills_since_last`` existed.
    res = run(bars)
    assert res.n_fills == 323 and res.n_places == 2730 and res.n_cancels == 2397
    assert res.n_decisions == 288 and res.round_trips == 77 and res.rejections == {}
    assert res.final_equity == pytest.approx(10250.088711700024, rel=1e-12)
    assert res.fees_paid == pytest.approx(50.23333829999999, rel=1e-12)
    assert res.gross_pnl == pytest.approx(299.2671900000007, rel=1e-12)
    eq = hashlib.sha256(np.asarray(res.daily_equity.values, dtype=float).tobytes()).hexdigest()
    fills = repr([(f["symbol"], f["side"], f["price"], f["qty"], f["client_id"]) for f in res.fills])
    assert eq.startswith("857a6278621adb52")
    assert hashlib.sha256(fills.encode()).hexdigest().startswith("89671c64e98622a5")


def test_explicit_none_strategy_is_the_default(bars):
    a, b = run(bars), run(bars, strategy=None, strategy_state=None)
    pd.testing.assert_series_equal(a.daily_equity, b.daily_equity)
    assert a.fills == b.fills


def test_fills_since_last_covers_every_fill_exactly_once(bars):
    seen: list[dict] = []

    def spy(snapshot, state):
        seen.extend(snapshot.fills_since_last)
        return PairedGridStrategy(PairedGridParams(levels=10, step=0.01), RISK)(snapshot, state)

    res = run(bars, strategy=spy)
    # fills of the last interval are only visible to the next decision, which never comes
    assert 0 < len(seen) <= res.n_fills
    assert [f["client_id"] for f in seen] == [f["client_id"] for f in res.fills[: len(seen)]]
    assert res.n_fills - len(seen) <= 6


def test_paired_grid_runs_on_synthetic_bars(bars):
    strat = PairedGridStrategy(PairedGridParams(levels=10, step=0.01), RISK)
    res = run(bars, strategy=strat)
    assert res.n_decisions == 3 * 96
    assert res.n_fills > 5 and res.round_trips >= 1
    assert res.n_taker_fills == 0
    assert res.rejections == {}
    assert res.max_exposure_total_post <= RISK.total_exposure_cap + 1e-9
    assert res.max_exposure_total_post > 0.05
    # the fills pair up: every sell closes quantity that a buy opened at a lower grid line
    held = {A: 0.0, B: 0.0}
    for f in res.fills:
        held[f["symbol"]] += f["qty"] if f["side"] == "buy" else -f["qty"]
        assert held[f["symbol"]] >= -1e-9


def test_rebalance_runs_on_synthetic_bars(bars):
    wide = run(bars, strategy=RebalanceStrategy(RebalanceParams(band=0.1), RISK))
    assert wide.n_fills >= 2  # at least the initial build-up
    strat = RebalanceStrategy(RebalanceParams(band=0.02), RISK)
    res = run(bars, strategy=strat)
    assert res.n_decisions == 3 * 96
    assert res.n_fills > 10 and res.n_fills > wide.n_fills  # a tighter band trades more
    assert {f["side"] for f in res.fills} == {"buy", "sell"}
    assert res.gross_pnl > 0 and res.round_trips == 0  # it never fully exits a coin
    assert res.n_taker_fills == 0
    assert res.rejections == {}
    assert res.max_exposure_total_post <= RISK.total_exposure_cap + 1e-9
    assert res.avg_exposure > 0.02


def test_strategy_runs_are_deterministic_and_picklable_through_run_many(bars):
    feats = precompute_features(bars, GRID.sigma_horizon_minutes)
    jobs = [
        ReplayJob(
            key=("v", i),
            kwargs=dict(
                grid=GRID, risk=RISK, start=START, end=END, decision_minutes=15,
                strategy=strat,
            ),
        )
        for i, strat in enumerate(
            [
                PairedGridStrategy(PairedGridParams(levels=10, step=0.01), RISK),
                RebalanceStrategy(RebalanceParams(band=0.2), RISK),
            ]
        )
    ]
    shared = dict(bars=bars, features=feats, filters=FILTERS)
    serial = run_many(jobs, processes=1, **shared)
    par = run_many(jobs, processes=2, **shared)
    for k in serial:
        pd.testing.assert_series_equal(serial[k].daily_equity, par[k].daily_equity)
        assert serial[k].n_fills == par[k].n_fills
