import numpy as np
import pandas as pd

from vv.core.trend import (
    TrendParams,
    apply_band,
    risk_dial,
    target_weights,
    trend_scores,
    vol_forecast,
)


def _panel(seed=0, n=400, k=4, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rets = rng.normal(0.0005, 0.03, size=(n, k))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=[f"C{i}" for i in range(k)])


def _features(n=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "stablecoin_supply_chg_30d": rng.normal(size=n),
        "vix_level": rng.normal(size=n),
        "global_ls_ratio": rng.normal(size=n),
    }, index=idx)


def test_no_lookahead():
    px = _panel()
    p = TrendParams()
    feats = _features()
    cut = 300
    full_dial = risk_dial(feats)
    w_full = target_weights(px, p, full_dial.reindex(px.index))
    px_t = px.iloc[:cut]
    w_t = target_weights(px_t, p, full_dial.reindex(px_t.index))
    pd.testing.assert_frame_equal(w_full.iloc[:cut], w_t)
    pd.testing.assert_frame_equal(trend_scores(px).iloc[:cut], trend_scores(px_t))
    pd.testing.assert_frame_equal(vol_forecast(px, 30).iloc[:cut], vol_forecast(px_t, 30))
    fcut = feats.iloc[:350]
    pd.testing.assert_series_equal(risk_dial(feats).iloc[:350], risk_dial(fcut))


def test_scores_monotone():
    idx = pd.date_range("2022-01-01", periods=200, freq="D", tz="UTC")
    px = pd.DataFrame({"UP": np.linspace(1, 2, 200), "DN": np.linspace(2, 1, 200)}, index=idx)
    s = trend_scores(px)
    assert s.iloc[:120].isna().all().all()
    assert (s["UP"].iloc[120:] == 1).all()
    assert (s["DN"].iloc[120:] == -1).all()


def test_scores_mixed():
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    # up over 120d and 60d, down over last 20d -> (1+1-1)/3
    v = np.concatenate([np.linspace(100, 200, n - 20), np.linspace(200, 190, 20)])
    s = trend_scores(pd.DataFrame({"X": v}, index=idx))
    assert np.isclose(s["X"].iloc[-1], 1 / 3)
    # down over 120d and 60d, up over last 20d -> -1/3
    v2 = np.concatenate([np.linspace(200, 100, n - 20), np.linspace(100, 110, 20)])
    s2 = trend_scores(pd.DataFrame({"X": v2}, index=idx))
    assert np.isclose(s2["X"].iloc[-1], -1 / 3)


def test_caps_property():
    for seed in range(200):
        rng = np.random.default_rng(seed)
        k = int(rng.integers(1, 8))
        px = _panel(seed=seed, n=200, k=k)
        px = px * rng.uniform(0.5, 2.0, size=k)
        for short in (False, True):
            w = target_weights(px, TrendParams(allow_short=short))
            assert (w.abs() <= 0.25 + 1e-12).all().all()
            assert (w.abs().sum(axis=1) <= 1.0 + 1e-12).all()


def test_spot_clamp():
    for seed in range(20):
        w = target_weights(_panel(seed=seed), TrendParams(allow_short=False))
        assert (w >= 0).all().all()
    # shorts do occur when allowed
    idx = pd.date_range("2022-01-01", periods=200, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    dn = pd.DataFrame({"A": 200 * np.exp(-np.arange(200) * 0.01 + rng.normal(0, 0.01, 200))},
                      index=idx)
    assert (target_weights(dn, TrendParams(allow_short=True)).iloc[-1] < 0).all()
    assert (target_weights(dn, TrendParams(allow_short=False)).iloc[-1] == 0).all()


def test_dial_all_penalties_hit_floor():
    n = 500
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    base = np.sin(np.arange(n))  # bounded noise
    f = pd.DataFrame({
        "stablecoin_supply_chg_30d": base.copy(),
        "vix_level": base.copy(),
        "global_ls_ratio": base.copy(),
    }, index=idx)
    f.iloc[-1] = [-10.0, 10.0, 10.0]
    d = risk_dial(f)
    assert d.iloc[-1] == 0.25
    assert ((d >= 0.25) & (d <= 1.0)).all()


def test_dial_missing_no_penalty():
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    f = pd.DataFrame(np.nan, index=idx,
                     columns=["stablecoin_supply_chg_30d", "vix_level", "global_ls_ratio"])
    assert (risk_dial(f) == 1.0).all()
    assert (risk_dial(f[["vix_level"]]) == 1.0).all()
    d = risk_dial(_features())
    assert ((d >= 0.25) & (d <= 1.0)).all()


def test_band():
    p = TrendParams()
    cur = pd.Series({"A": 0.10, "B": 0.0, "C": 0.10, "D": 0.0})
    tgt = pd.Series({"A": 0.11, "B": 0.02, "C": 0.20, "D": 0.01})
    out = apply_band(cur, tgt, p)
    assert out["A"] == 0.10  # 0.01 < band 0.02775? band=max(.02,.0275)
    assert out["B"] == 0.02  # 0 -> 0.02 executes
    assert out["C"] == 0.20  # big change executes
    assert out["D"] == 0.0  # below 0.02 ignored
    out2 = apply_band(pd.Series({"A": 0.0}), pd.Series({"A": 0.25}), p)
    assert out2["A"] == 0.25


def test_eligibility():
    px = _panel(n=300, k=2)
    px.loc[px.index[:200], "C1"] = np.nan  # C1 has history of 100 days at the end
    w = target_weights(px, TrendParams(allow_short=True))
    assert (w["C1"] == 0).all()
    assert (w["C0"].iloc[121:] != 0).any()
    assert (w.iloc[:120] == 0).all().all()
