"""Tests for research/feature_screen.py (pre-registered external-data feature screen)."""
import numpy as np
import pandas as pd
import pytest

from research import feature_screen as fs

UTC = "UTC"


def _cal(n=500, start="2022-01-03"):
    return pd.date_range(start, periods=n, freq="1D", tz=UTC, name="date")


def _synthetic_inputs(n=500, seed=0):
    rng = np.random.default_rng(seed)
    cal = _cal(n)
    prices = {}
    for i, s in enumerate(fs.SYMS):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        prices[s] = pd.DataFrame({"close": c, "volume": rng.uniform(1e3, 2e3, n)}, index=cal)
    bd = cal[cal.day_of_week < 5]
    fred = pd.DataFrame({c: 100 + np.cumsum(rng.normal(0, 1, len(bd))) for c in ["DTWEXBGS", "DGS10", "VIXCLS", "NASDAQCOM",
                                                                            "RRPONTSYD"]}, index=bd)
    wed = cal[cal.day_of_week == 2]
    fred = fred.reindex(cal)
    fred.loc[wed, "WALCL"] = 8e6 + np.cumsum(rng.normal(0, 1e4, len(wed)))
    fred.loc[wed, "WTREGEN"] = 700 + np.cumsum(rng.normal(0, 5, len(wed)))
    metrics = {s: pd.DataFrame({c: rng.uniform(1, 2, n) for c in [
        "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]}, index=cal)
        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}
    premium = {s: pd.DataFrame({"perp_premium": rng.normal(0, 1e-3, n)}, index=cal) for s in metrics}
    funding = {s: pd.DataFrame({"funding_rate": rng.normal(1e-4, 1e-4, n)}, index=cal) for s in fs.SYMS}
    return {"prices": prices, "fred": fred,
            "fear_greed": pd.DataFrame({"fear_greed": rng.uniform(0, 100, n)}, index=cal),
            "stablecoins": pd.DataFrame({"stablecoin_supply_usd": 1e11 + np.cumsum(rng.normal(1e8, 1e8, n))}, index=cal),
            "metrics": metrics, "premium": premium, "funding": funding}


# ------------------------------------------------------------------ lagging
def test_fred_daily_lagged_one_business_day():
    cal = _cal(30)
    s = pd.Series(np.arange(30, dtype=float), index=cal)
    s = s[s.index.day_of_week < 5]  # FRED has business-day observations only
    out = fs.lag_to_calendar(s, cal, "bday1")
    fri = pd.Timestamp("2022-01-07", tz=UTC)  # observation dated Friday
    assert out[fri] == s[fri - pd.Timedelta(days=1)]  # Fri row still holds Thursday's obs
    assert out[pd.Timestamp("2022-01-10", tz=UTC)] == s[fri]  # Fri obs first usable Monday


def test_weekly_wednesday_observation_available_thursday():
    cal = _cal(30)
    s = pd.Series(np.nan, index=cal)
    wed = pd.Timestamp("2022-01-12", tz=UTC)
    s[wed] = 5.0
    out = fs.lag_to_calendar(s, cal, "day1")
    assert np.isnan(out[wed])
    assert out[wed + pd.Timedelta(days=1)] == 5.0


def test_fed_liquidity_uses_lagged_components():
    inp = _synthetic_inputs()
    cal = _cal()
    m = fs.macro_features(cal, inp["fred"])
    fred = inp["fred"]
    d = pd.Timestamp("2022-06-23", tz=UTC)  # a Thursday
    walcl = fred["WALCL"].dropna()
    walcl_known = walcl[walcl.index <= d - pd.Timedelta(days=1)].iloc[-1] / 1000
    assert walcl_known > 0
    # changing the Wednesday-after observation must not alter the Thursday-before value
    f2 = fred.copy()
    f2.loc[pd.Timestamp("2022-06-22", tz=UTC), "WALCL"] = 1e9
    m2 = fs.macro_features(cal, f2)
    assert m["fed_liquidity_chg_4w"][d - pd.Timedelta(days=1)] == m2["fed_liquidity_chg_4w"][d - pd.Timedelta(days=1)]
    assert m["fed_liquidity_chg_4w"][d] != m2["fed_liquidity_chg_4w"][d]


def test_no_lookahead_for_real_feature_builder():
    inp = _synthetic_inputs()
    cal = _cal()
    t0 = cal[300]

    def build(i):
        f = fs.build_features(i, cal)
        return pd.concat([f["single"], f["basket"].add_prefix("b_")], axis=1)

    assert fs.check_no_lookahead(build, inp, t0)


def test_leaked_feature_is_caught():
    inp = _synthetic_inputs()
    cal = _cal()

    def leaky(i):  # next-day return used as a "feature"
        c = i["prices"]["BTCUSDT"]["close"]
        return pd.DataFrame({"leak": np.log(c).shift(-1) - np.log(c)})

    assert not fs.check_no_lookahead(leaky, inp, cal[300])


def test_features_have_prereg_columns():
    inp = _synthetic_inputs()
    cal = _cal()
    f = fs.build_features(inp, cal)
    for u in ("single", "basket"):
        assert list(f[u].columns) == fs.ALL_FEATURES
        assert f[u].iloc[-1].notna().all(), f[u].iloc[-1][f[u].iloc[-1].isna()]


# ------------------------------------------------------------------ targets
def test_targets_forward_looking():
    close = pd.Series([100, 100, 95, 85, 90, 100, 100, 100, 100, 100, 100, 100.0], index=_cal(12))
    t = fs.build_targets(close)
    assert t["ret_1d"].iloc[0] == pytest.approx(0.0)
    assert t["ret_1d"].iloc[1] == pytest.approx(np.log(0.95))
    assert t["ret_7d"].iloc[1] == pytest.approx(0.0)
    assert t["crash_7d"].iloc[1] == 1.0  # min 85 vs 100
    assert t["crash_7d"].iloc[0] == 1.0
    assert t["crash_7d"].iloc[2] == 1.0  # base 95: min 85/95-1 = -10.5% -> crash
    assert t["crash_7d"].iloc[4] == 0.0  # base 90: nothing below
    assert np.isnan(t["crash_7d"].iloc[5])  # fewer than 7 future days


# ------------------------------------------------------------------ BH-FDR
def test_bh_fdr_hand_example():
    # m=5, q=0.10: thresholds 0.02 0.04 0.06 0.08 0.10; sorted p = .001 .03 .035 .09 .5
    # largest k with p_k <= thr_k: k=3 (0.035<=0.06); k=4 0.09>0.08 -> reject first three
    p = np.array([0.5, 0.001, 0.09, 0.035, 0.03])
    assert fs.bh_fdr(p, 0.10).tolist() == [False, True, False, True, True]


def test_bh_fdr_step_up_and_nan():
    # sorted .04 .05 .9 vs thresholds .0333 .0667 .1: p_1 fails alone but k=2 passes, so both rejected (step-up)
    assert fs.bh_fdr([0.05, 0.04, 0.9], 0.10).tolist() == [True, True, False]
    assert fs.bh_fdr([0.01, 0.015, np.nan, 0.5], 0.10).tolist() == [True, True, False, False]
    assert not fs.bh_fdr([np.nan, np.nan]).any()


# ------------------------------------------------------------------ walk-forward
def test_walk_forward_never_trains_on_test_rows():
    idx = _cal(1500, "2021-01-01")
    n = 0
    for h in (1, 7):
        for tr, te in fs.walk_forward_splits(idx, pd.Timestamp("2024-01-01", tz=UTC), pd.Timestamp("2024-12-31", tz=UTC), h):
            n += 1
            assert set(tr).isdisjoint(te)
            assert idx[tr].max() + pd.Timedelta(days=h) < idx[te].min()  # train target window ends before test month
            assert idx[te].min() >= pd.Timestamp("2024-01-01", tz=UTC)
    assert n >= 20


def test_walk_forward_predictions_change_only_with_past_data():
    rng = np.random.default_rng(1)
    idx = _cal(1500, "2021-01-01")
    X = pd.DataFrame(rng.normal(size=(1500, 3)), index=idx, columns=list("abc"))
    y = pd.Series(0.5 * X["a"].shift(1).fillna(0) + rng.normal(0, 1, 1500), index=idx)
    start, end = pd.Timestamp("2024-06-01", tz=UTC), pd.Timestamp("2024-09-30", tz=UTC)
    r1 = fs.walk_forward(X, y, "ridge", 1, start, end)
    y2 = y.copy()
    y2[y2.index >= pd.Timestamp("2024-08-01", tz=UTC)] += 100  # corrupt targets from August on
    r2 = fs.walk_forward(X, y2, "ridge", 1, start, end)
    jun_jul = r1.index < pd.Timestamp("2024-08-01", tz=UTC)
    np.testing.assert_allclose(r1.pred[jun_jul], r2.pred[jun_jul])


def test_ridge_and_logit_learn_signal():
    rng = np.random.default_rng(2)
    n = 2000
    idx = _cal(n, "2019-01-01")
    X = pd.DataFrame(rng.normal(size=(n, 2)), index=idx, columns=["a", "b"])
    y = pd.Series(X["a"] + rng.normal(0, 1, n), index=idx)
    s, e = pd.Timestamp("2023-01-01", tz=UTC), idx[-1]
    r = fs.walk_forward(X, y, "ridge", 1, s, e)
    assert fs.oos_metrics(r, "ridge")["r2"] > 0.3
    yb = pd.Series((X["a"] + rng.normal(0, 1, n) > 1).astype(float), index=idx)
    rb = fs.walk_forward(X, yb, "logit", 1, s, e)
    assert fs.oos_metrics(rb, "logit")["auc"] > 0.75


def test_auc_hand_example():
    assert fs.auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


# ------------------------------------------------------------------ HAC
def test_hac_deterministic_and_matches_ols_at_zero_lags():
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    y = 0.3 * x + rng.normal(size=300)
    a = fs.hac_tstat(x, y, 7)
    assert a == fs.hac_tstat(x, y, 7)
    b, t0 = fs.hac_tstat(x, y, 0)
    # White (HC0) t for slope
    X = np.column_stack([np.ones(300), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    V = np.linalg.inv(X.T @ X) @ (X.T * e**2 @ X) @ np.linalg.inv(X.T @ X)
    assert b == pytest.approx(beta[1])
    assert t0 == pytest.approx(beta[1] / np.sqrt(V[1, 1]))


def test_hac_larger_than_naive_for_overlapping_returns():
    rng = np.random.default_rng(4)
    n = 1500
    x = pd.Series(rng.normal(size=n))
    r = pd.Series(rng.normal(size=n + 7))
    y = r.rolling(7).sum().shift(-7).iloc[:n]  # overlapping 7d sums: autocorrelated, unrelated to x
    ok = y.notna()
    _, t_naive = fs.hac_tstat(x[ok], y[ok], 0)
    _, t_hac = fs.hac_tstat(x[ok], y[ok], 7)
    assert t_naive != t_hac


def test_ic_test_detects_signal_and_is_deterministic():
    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(size=500))
    y = x * 0.5 + rng.normal(size=500)
    r = fs.ic_test(x, y, 7)
    assert r["ic"] > 0.3 and r["p"] < 1e-6
    assert r == fs.ic_test(x, y, 7)


# ------------------------------------------------------------------ event study / decision
def test_top_drawdowns_non_overlapping_and_forced():
    rng = np.random.default_rng(6)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 1000))), index=_cal(1000))
    forced = close.index[-9]
    ev = fs.top_drawdowns(close, 8, force=forced)
    assert len(ev) == 8 and forced in set(ev.base_day)
    days = sorted(ev.base_day)
    assert all((b - a).days > 7 for a, b in zip(days, days[1:], strict=False))


def test_decision_rule_needs_both_criteria():
    uni = pd.DataFrame([{"group": "B", "fdr_pass": True, "same_sign": True, "universe": "single", "feature": "f",
                         "target": "crash_7d"},
                        {"group": "C", "fdr_pass": False, "same_sign": True, "universe": "single", "feature": "g",
                         "target": "ret_1d"},
                        {"group": "D", "fdr_pass": True, "same_sign": False, "universe": "single", "feature": "h",
                         "target": "ret_1d"}])
    inc = pd.DataFrame([{"group": g, "universe": "single", "target": "crash_7d", "delta": 0.05} for g in "BCD"])
    d = fs.decide(uni, inc).set_index("group")["verdict"]
    assert d["B"] == "contributes"
    assert d["C"] == d["D"] == "does not contribute"
    inc["delta"] = 0.01  # below AUC threshold
    assert fs.decide(uni, inc).set_index("group")["verdict"]["B"] == "does not contribute"


def test_end_to_end_synthetic_runs():
    inp = _synthetic_inputs(n=1500)
    cal, feats, targets = fs.build_all(inp, last_day=inp["prices"]["BTCUSDT"].index[-1])
    assert set(feats) == {"single", "basket"}
    ev = fs.event_study(feats["single"], fs.top_drawdowns(inp["prices"]["BTCUSDT"]["close"], 8))
    assert len(ev) == len(fs.ALL_FEATURES)
