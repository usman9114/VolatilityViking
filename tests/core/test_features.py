import time

import numpy as np
import pandas as pd
import pytest

from vv.core.features import compute_features, latest_features
from vv.core.types import SymbolFeatures


def _gbm(n_minutes: int, sigma_1m: float = 0.0005, drift: float = 0.0, seed: int = 0,
         start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, sigma_1m, n_minutes)
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx
    )


def test_columns_and_index():
    bars = _gbm(2000)
    f = compute_features(bars, horizon_minutes=60, ema_span_4h=5)
    assert list(f.columns) == ["close", "sigma_h", "ema_4h", "regime_ok", "warm"]
    assert f.index.equals(bars.index)
    assert f["warm"].dtype == bool and f["regime_ok"].dtype == bool


def test_warm_false_at_start_and_true_later():
    bars = _gbm(60 * 24 * 40)
    f = compute_features(bars, horizon_minutes=60)
    assert not f["warm"].iloc[0]
    assert not f["warm"].iloc[:1000].any()
    assert not f["regime_ok"][~f["warm"]].any()
    assert f["warm"].iloc[-1]
    # 200 completed 4h bars = 800h of history needed
    first_warm = f.index[f["warm"]][0]
    assert first_warm - f.index[0] >= pd.Timedelta(hours=800)


def test_no_lookahead_truncate_and_recompute():
    bars = _gbm(60 * 24 * 12, seed=3)
    full = compute_features(bars, horizon_minutes=60, ema_span_4h=10)
    for cut in (3000, 7777, 12345):
        part = compute_features(bars.iloc[:cut], horizon_minutes=60, ema_span_4h=10)
        pd.testing.assert_frame_equal(part, full.iloc[:cut], check_exact=False, rtol=1e-9)


def test_future_change_does_not_alter_past():
    bars = _gbm(60 * 24 * 12, seed=4)
    base = compute_features(bars, 60, 10)
    mutated = bars.copy()
    mutated.iloc[9000:, mutated.columns.get_loc("close")] *= 3.0
    after = compute_features(mutated, 60, 10)
    pd.testing.assert_frame_equal(base.iloc[:9000], after.iloc[:9000])


@pytest.mark.parametrize("horizon", [30, 60, 240])
def test_sigma_h_matches_gbm_truth(horizon):
    sigma_1m = 0.0005
    bars = _gbm(60 * 24 * 90, sigma_1m=sigma_1m, seed=11)
    f = compute_features(bars, horizon_minutes=horizon, ema_span_4h=10)
    truth = sigma_1m * np.sqrt(horizon)
    tail = f["sigma_h"].iloc[-60 * 24 * 30 :]
    assert tail.notna().all()
    assert abs(tail.mean() / truth - 1) < 0.15
    assert abs(f["sigma_h"].iloc[-1] / truth - 1) < 0.15


def test_ema_4h_ignores_in_progress_bar():
    # 30 full 4h bars plus 100 minutes of a 31st
    n = 30 * 240 + 100
    bars = _gbm(n, seed=5, start="2024-01-01 00:00")
    span = 5
    f = compute_features(bars, 60, ema_span_4h=span)

    closes4 = bars["close"].resample("4h").last()
    done = closes4.iloc[:30]  # completed bars
    expected = done.ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1]
    assert f["ema_4h"].iloc[-1] == pytest.approx(expected)
    assert f["ema_4h"].iloc[-100] == pytest.approx(expected)  # first minute of in-progress bar

    # wildly different prices inside the in-progress bar leave the EMA untouched
    mutated = bars.copy()
    mutated.iloc[-100:, mutated.columns.get_loc("close")] *= 10
    g = compute_features(mutated, 60, ema_span_4h=span)
    pd.testing.assert_series_equal(f["ema_4h"], g["ema_4h"])
    # ... but the regime flag reacts to the live price
    assert g["regime_ok"].iloc[-1]


def test_regime_ok_is_close_vs_ema():
    up = compute_features(_gbm(60 * 24 * 40, drift=0.0002, seed=1), 60, 20)
    down = compute_features(_gbm(60 * 24 * 40, drift=-0.0002, seed=1), 60, 20)
    assert up["regime_ok"].iloc[-1] and not down["regime_ok"].iloc[-1]
    w = up[up["warm"]]
    assert (w["regime_ok"] == (w["close"] >= w["ema_4h"])).all()


def test_latest_features_matches_last_row():
    bars = _gbm(60 * 24 * 40, seed=2)
    f = compute_features(bars, 60)
    lf = latest_features(bars, 60)
    assert isinstance(lf, SymbolFeatures)
    last = f.iloc[-1]
    assert lf.close == last["close"] and lf.sigma_h == pytest.approx(last["sigma_h"])
    assert lf.ema_4h == pytest.approx(last["ema_4h"])
    assert lf.regime_ok == bool(last["regime_ok"]) and lf.warm is True


def test_latest_features_cold():
    lf = latest_features(_gbm(500), 60)
    assert lf.warm is False and lf.regime_ok is False


def test_rejects_unsorted_index():
    bars = _gbm(100).iloc[::-1]
    with pytest.raises(ValueError):
        compute_features(bars, 60)


def test_runs_on_naive_index_with_gaps():
    bars = _gbm(60 * 24 * 40, seed=7)
    bars.index = bars.index.tz_localize(None)
    bars = bars.drop(bars.index[5000:5030])
    f = compute_features(bars, 60, ema_span_4h=20)
    assert len(f) == len(bars) and f["warm"].iloc[-1]


def test_vectorised_speed():
    bars = _gbm(60 * 24 * 365)  # ~525k rows
    t = time.perf_counter()
    compute_features(bars, 60)
    assert time.perf_counter() - t < 5.0
