"""Metrics on hand-computed series."""

import math

import numpy as np
import pandas as pd
import pytest

from vv.sim import metrics as m


def daily(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_sortino_uses_downside_deviation_not_std_of_negatives():
    # equity 100 -> daily returns +10%, -5%, +8%, -2% (hand-built from returns)
    r = np.array([0.10, -0.05, 0.08, -0.02])
    eq = daily(100 * np.concatenate([[1.0], np.cumprod(1 + r)]))
    dd = math.sqrt((0.0**2 + 0.05**2 + 0.0**2 + 0.02**2) / 4)  # mean(min(r,0)^2) over ALL r
    assert dd == pytest.approx(math.sqrt(0.000725))
    expected = r.mean() / dd * math.sqrt(365)
    assert m.sortino(m.daily_returns(eq)) == pytest.approx(expected)
    # not the sample std of the negative returns only, and not the sharpe
    wrong = r.mean() / np.std(r[r < 0], ddof=1) * math.sqrt(365)
    assert m.sortino(m.daily_returns(eq)) != pytest.approx(wrong)


def test_sortino_no_losses_is_inf_and_flat_is_zero():
    assert m.sortino(m.daily_returns(daily([100, 101, 102]))) == math.inf
    assert m.sortino(m.daily_returns(daily([100, 100, 100]))) == 0.0


def test_sharpe_hand_computed_and_flat_series():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    eq = daily(100 * np.concatenate([[1.0], np.cumprod(1 + r)]))
    expected = r.mean() / r.std(ddof=1) * math.sqrt(365)
    assert m.sharpe(m.daily_returns(eq)) == pytest.approx(expected)
    assert m.sharpe(m.daily_returns(daily([5, 5, 5, 5]))) == 0.0


def test_max_drawdown_and_time_under_water_on_known_curve():
    eq = daily([100, 110, 105, 99, 104, 110, 120, 90, 100])
    # peak 110 (day1) -> trough 99 (day3): -10%; later 120 -> 90 is -25%
    assert m.max_drawdown(eq) == pytest.approx(0.25)
    # episodes: day1 peak -> recovered day5 (110 >= 110) = 4 days; day6... peak 120 at day6,
    # never recovered by day8 => 2 days
    assert m.longest_underwater_days(eq) == pytest.approx(4.0)
    # monotone rising: no drawdown, nothing under water
    up = daily([100, 101, 102, 103])
    assert m.max_drawdown(up) == 0.0
    assert m.longest_underwater_days(up) == 0.0


def test_underwater_counts_open_ended_episode_to_series_end():
    eq = daily([100, 90, 95, 96, 97])  # peak day0, never recovered: 4 days
    assert m.longest_underwater_days(eq) == pytest.approx(4.0)
    assert m.max_drawdown(eq) == pytest.approx(0.10)


def test_total_return_cagr_vol():
    eq = daily([100.0] + [100.0] * 364 + [121.0])  # 365 days
    assert m.total_return(eq) == pytest.approx(0.21)
    assert m.cagr(eq) == pytest.approx(0.21)  # exactly one 365-day year
    r = m.daily_returns(eq)
    assert m.ann_vol(r) == pytest.approx(float(r.std(ddof=1)) * math.sqrt(365))


def test_sharpe_ci_deterministic_under_seed_and_brackets_point():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 400))
    a = m.sharpe_ci(r, mean_block=10, reps=500, seed=7)
    b = m.sharpe_ci(r, mean_block=10, reps=500, seed=7)
    c = m.sharpe_ci(r, mean_block=10, reps=500, seed=8)
    assert a == b
    assert a != c
    assert a[0] < m.sharpe(r) < a[1]


def test_sharpe_ci_positive_drift_excludes_zero_and_zero_drift_does_not():
    rng = np.random.default_rng(3)
    strong = pd.Series(rng.normal(0.01, 0.01, 500))
    lo, _ = m.sharpe_ci(strong, reps=500, seed=1)
    assert lo > 0
    noise = pd.Series(rng.normal(0.0, 0.01, 500))
    lo, hi = m.sharpe_ci(noise, reps=500, seed=1)
    assert lo < 0 < hi


def test_stationary_indices_are_valid_and_blocky():
    idx = m.stationary_bootstrap_indices(50, 10.0, 20, np.random.default_rng(0))
    assert idx.shape == (20, 50)
    assert idx.min() >= 0 and idx.max() < 50
    steps = np.diff(idx, axis=1) % 50
    # consecutive-step fraction should be close to 1 - 1/mean_block
    assert 0.8 < (steps == 1).mean() < 0.97


def test_net_capture_over_fees():
    assert m.net_capture_over_fees(30.0, 10.0) == pytest.approx(3.0)
    assert m.net_capture_over_fees(-5.0, 10.0) == pytest.approx(-0.5)
    assert m.net_capture_over_fees(5.0, 0.0) == 0.0  # nothing traded


def test_baselines_cash_equal_weight_and_scaled():
    idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
    a = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [10.0, 10.0, 20.0]}, index=idx)
    b = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [10.0, 10.0, 5.0]}, index=idx)
    # marks at t are the close of the bar labelled t-1min
    times = pd.DatetimeIndex([idx[0] + pd.Timedelta("1min"), idx[0] + pd.Timedelta("3min")])
    out = m.baseline_curves({"A": a, "B": b}, times, 1000.0, avg_exposure=0.5)
    assert list(out["cash"]) == [1000.0, 1000.0]
    # start marks (10, 10); end marks (20, 5): equal weight 500/500 -> 1000 + 500 (A x2) + 250 (B x0.5) -> 1250
    assert list(out["equal_weight_bh"]) == pytest.approx([1000.0, 1250.0])
    # half in the basket, half in cash
    assert list(out["bh_scaled_to_exposure"]) == pytest.approx([1000.0, 1125.0])


def test_summarize_fields():
    eq = daily([100, 101, 99, 103])
    s = m.summarize(eq, fees_paid=2.0, gross_pnl=5.0, round_trips=3, avg_exposure=0.2)
    assert s["total_return"] == pytest.approx(0.03)
    assert s["net_capture_over_fees"] == pytest.approx(2.5)
    assert s["round_trips"] == 3
    assert s["avg_exposure"] == 0.2
    assert set(s) >= {"cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "longest_underwater_days"}
