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
             "--data-dir", str(tmp_path), "--reps", "20"])
    text = out.read_text()
    assert "Verdict" in text and ("FAIL" in text or "PASS" in text)
