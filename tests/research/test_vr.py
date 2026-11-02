"""Tests for research/g0_variance_ratio.py (gate G0 edge pre-check)."""
import numpy as np
import pandas as pd
import pytest

from research import g0_variance_ratio as g0


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * 1e-3
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return r


def test_ar1_negative_autocorr_vr_below_one_ci_excludes_one():
    r = _ar1(-0.2, 40_000, seed=1)
    res = g0.variance_ratio(r, 2)
    assert res.vr < 1
    assert abs(res.vr - 0.8) < 0.03  # theory: 1 + phi
    ci = g0.vr_bootstrap_ci(r, [2], mean_block=50, reps=500, seed=7)
    assert ci[0, 1] < 1.0


def test_gbm_ci_contains_one():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 1e-3, 40_000)  # log returns of a GBM
    ci = g0.vr_bootstrap_ci(r, [2, 12], mean_block=50, reps=500, seed=7)
    assert np.all(ci[:, 0] < 1.0) and np.all(ci[:, 1] > 1.0)
    res = g0.variance_ratio(r, 12)
    assert res.p > 0.01


def test_block_bootstrap_deterministic_under_seed():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 1e-3, 5_000)
    a = g0.vr_bootstrap_ci(r, [2, 5], mean_block=40, reps=100, seed=11)
    b = g0.vr_bootstrap_ci(r, [2, 5], mean_block=40, reps=100, seed=11)
    c = g0.vr_bootstrap_ci(r, [2, 5], mean_block=40, reps=100, seed=12)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    i1 = g0.stationary_bootstrap_indices(1000, 40, np.random.default_rng(1))
    i2 = g0.stationary_bootstrap_indices(1000, 40, np.random.default_rng(1))
    np.testing.assert_array_equal(i1, i2)
    assert i1.min() >= 0 and i1.max() < 1000


def test_dfa_white_noise_hurst_about_half():
    rng = np.random.default_rng(9)
    r = rng.standard_normal(60_000)
    h = g0.dfa_hurst(r, g0.dfa_scales(12, 288))
    assert abs(h - 0.5) < 0.1


def test_dfa_persistent_series_above_half():
    r = _ar1(0.5, 60_000, seed=2)  # short memory, but positively correlated at small scales
    h = g0.dfa_hurst(r, g0.dfa_scales(4, 40))
    assert h > 0.55


def test_pipeline_smoke_on_synthetic_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    idx = pd.date_range("2024-01-01", periods=60 * 24 * 40, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(idx))))
    df = pd.DataFrame(
        {"open": close, "high": close * 1.0005, "low": close * 0.9995,
         "close": close, "volume": 1.0}, index=idx)
    (tmp_path / "klines").mkdir()
    df.to_parquet(tmp_path / "klines" / "TESTUSDT_1m.parquet")
    out = tmp_path / "out.md"
    g0.main(["--symbols", "TESTUSDT", "--out", str(out),
             "--data-dir", str(tmp_path), "--reps", "20",
             "--exclude-listing-days", "5"])
    text = out.read_text()
    assert "Verdict" in text and ("FAIL" in text or "PASS" in text)
    assert "Final (both)" in text and "Robust" in text


# --------------------------------------------------------------------------
# G0-robust additions (WP-S0D)
# --------------------------------------------------------------------------
def _ar1_series(phi: float, n: int, seed: int) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(_ar1(phi, n, seed), index=idx)


def test_winsorise_caps_spike_and_moves_vr_back():
    clean = _ar1_series(-0.3, 20_000, seed=4)
    spiked = clean.copy()
    sigma = 1.4826 * float(np.median(np.abs(clean - clean.median())))
    for pos, sign in [(3_000, 1), (9_000, -1), (15_000, 1)]:
        spiked.iloc[pos] = sign * 20 * sigma
    wins, n_capped = g0.winsorise_monthly(spiked, k=5.0)

    assert n_capped >= 3
    assert wins.abs().max() < 6 * sigma          # spike capped near 5 sigma
    assert wins.iloc[3_000] > 0 and wins.iloc[9_000] < 0  # sign preserved
    vr = lambda s: g0.variance_ratio(s.values, 12).vr
    assert abs(vr(wins) - vr(clean)) < abs(vr(spiked) - vr(clean))
    # input untouched, non-spike bars mostly unchanged
    assert spiked.abs().max() > 15 * sigma
    assert (wins != spiked).sum() == n_capped


def test_winsorise_scale_is_per_calendar_month():
    idx = pd.date_range("2024-01-01", "2024-02-29 23:55", freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    vol = np.where(idx.month == 1, 1e-4, 1e-3)
    r = pd.Series(rng.standard_normal(len(idx)) * vol, index=idx)
    r.iloc[100] = 1e-2   # 100 sigma in a quiet January, only 10 sigma-ish in February scale
    wins, _ = g0.winsorise_monthly(r, k=5.0)
    assert wins.iloc[100] < 1e-3                       # capped by January's own scale
    assert wins[idx.month == 2].abs().max() > 3e-3     # February not capped at January's scale


def test_exclude_window_removes_exactly_those_bars():
    idx = pd.date_range("2025-10-01", "2025-10-20", freq="5min", tz="UTC", inclusive="left")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    start, end = pd.Timestamp("2025-10-10", tz="UTC"), pd.Timestamp("2025-10-12", tz="UTC")
    out = g0.exclude_windows(s, [(start, end)])
    assert len(s) - len(out) == 2 * 288
    assert not ((out.index >= start) & (out.index < end)).any()
    kept = s[(s.index < start) | (s.index >= end)]
    pd.testing.assert_series_equal(out, kept)
    assert pd.Timestamp("2025-10-09 23:55", tz="UTC") in out.index
    assert end in out.index and start not in out.index


def test_parse_window_and_listing_drop():
    w = g0.parse_windows([["2025-10-10", "2025-10-12"]])
    assert w == [(pd.Timestamp("2025-10-10", tz="UTC"), pd.Timestamp("2025-10-12", tz="UTC"))]
    idx = pd.date_range("2024-01-01", periods=60 * 24 * 40, freq="1min", tz="UTC")
    df = pd.DataFrame({"close": 1.0}, index=idx)
    out = g0.drop_listing_days(df, 30)
    assert out.index[0] == pd.Timestamp("2024-01-31", tz="UTC")
    assert len(g0.drop_listing_days(df, 0)) == len(df)


def test_holm_matches_hand_computation():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005,0.01,0.03,0.04 -> x4,x3,x2,x1 = 0.02,0.03,0.06,0.04 -> cummax 0.02,0.03,0.06,0.06
    adj = g0.holm_adjust(p)
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(g0.holm_adjust([0.5, 0.9]), [1.0, 1.0])  # 0.5*2=1.0; 0.9*1 -> cummax 1.0
    np.testing.assert_allclose(g0.holm_adjust([0.2]), [0.2])


def test_recentred_ci_always_contains_point_estimate():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(150, 600))                     # short sample
        r = rng.standard_t(2.5, n) * 1e-3 * rng.choice([1, 1, 8], n)  # heavy tails
        qs = [2, 12, 48]
        ci = g0.vr_bootstrap_ci(r, qs, mean_block=20, reps=200, seed=seed)
        vrs = g0._vr_point(r, qs)
        assert np.all(ci[:, 0] <= vrs) and np.all(vrs <= ci[:, 1])
