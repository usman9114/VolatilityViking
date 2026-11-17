"""Portfolio simulator on hand-computed paths.

Spot bars are labelled by close time (price at the label instant); perp bars by open time.
"""

import numpy as np
import pandas as pd
import pytest

from vv.core.trend import TrendParams, apply_band
from vv.sim.portfolio_sim import simulate

T0 = pd.Timestamp("2024-01-01")  # decision row for day 2024-01-01 = close at 2024-01-02 00:00


def spot_bars(prices: dict[str, float], days=5) -> pd.DataFrame:
    idx = pd.date_range(T0, periods=days * 24 + 1, freq="h")
    s = pd.Series(100.0, index=idx)
    for ts, p in prices.items():
        s[pd.Timestamp(ts):] = p
    return pd.DataFrame({"close": s})


def perp_bars(open_px: dict[str, float], days=5, wick=None) -> pd.DataFrame:
    """Constant price path; bar opened at T has open=close=price at T. ``wick``: {open_ts: (hi, lo)}."""
    idx = pd.date_range(T0, periods=days * 24, freq="h")
    p = pd.Series(100.0, index=idx)
    for ts, v in open_px.items():
        p[pd.Timestamp(ts):] = v
    df = pd.DataFrame({"open": p, "high": p, "low": p, "close": p})
    # close of bar T equals open of bar T+1 for a continuous path
    df["close"] = df["open"].shift(-1).fillna(df["open"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    for ts, (hi, lo) in (wick or {}).items():
        df.loc[pd.Timestamp(ts), ["high", "low"]] = (hi, lo)
    return df


def w(vals, col="X"):
    return pd.DataFrame({col: vals}, index=pd.date_range(T0, periods=len(vals), freq="D"))


def test_spot_buy_next_bar_close_with_fee_and_slippage():
    bars = spot_bars({"2024-01-02 01:00": 100.0, "2024-01-03 00:00": 110.0})
    res = simulate(w([0.5]), "spot", {"X": bars}, {}, fee=0.001, slippage_bps=10)
    qty = 0.5 * 10_000 / 100.0  # sized at the execution price (equity 10000)
    cost = qty * 100.0 * 1.001 * 1.001
    assert res.trades.iloc[0]["fee"] == pytest.approx(qty * 100.0 * 1.001 * 0.001)
    cash = 10_000 - cost
    assert res.equity_hourly.loc["2024-01-02 01:00"] == pytest.approx(cash + qty * 100.0)
    assert res.equity_hourly.loc["2024-01-03 00:00"] == pytest.approx(cash + qty * 110.0)
    tr = res.trades.iloc[0]
    assert tr["time"] == pd.Timestamp("2024-01-02 01:00")
    assert tr["qty"] == pytest.approx(qty)
    assert res.max_gross == pytest.approx(qty * 110.0 / (cash + qty * 110.0))


def test_spot_negative_weight_raises():
    with pytest.raises(ValueError):
        simulate(w([0.5, -0.1]), "spot", {"X": spot_bars({})}, {}, fee=0.001, slippage_bps=5)
    with pytest.raises(ValueError):
        simulate(w([1.2]), "spot", {"X": spot_bars({})}, {}, fee=0.001, slippage_bps=5)


@pytest.mark.parametrize("venue", ["spot", "perp"])
def test_execution_strictly_after_decision(venue):
    bars = spot_bars({}) if venue == "spot" else perp_bars({})
    res = simulate(w([0.4, 0.2]), venue, {"X": bars}, {}, fee=0.0005, slippage_bps=5)
    decision_close = pd.Timestamp("2024-01-02 00:00")  # close of day 2024-01-01
    first = res.trades.iloc[0]["time"]
    assert first > decision_close and first == decision_close + pd.Timedelta(hours=1)
    second = res.trades[res.trades["time"] > first].iloc[0]["time"]
    assert second == pd.Timestamp("2024-01-03 01:00")


def test_perp_short_gains_when_price_falls():
    bars = perp_bars({"2024-01-02 01:00": 100.0, "2024-01-03 00:00": 90.0})
    res = simulate(w([-0.5]), "perp", {"X": bars}, {}, fee=0.0005, slippage_bps=10)
    qty = -0.5 * 10_000 / 100.0  # -50 contracts, filled at the bar open
    tp = 100.0 * (1 - 0.001)  # a sell pays -slippage
    fee = 0.0005 * 50 * tp
    cash = 10_000 + 50 * tp - fee
    assert res.trades.iloc[0]["fee"] == pytest.approx(fee)
    assert res.equity_hourly.loc["2024-01-03 01:00"] == pytest.approx(cash + qty * 90.0)
    assert res.equity_hourly.loc["2024-01-03 01:00"] > 10_000
    assert res.liquidations == []


def test_perp_funding_sign_short_receives_long_pays():
    funding = {"X": pd.Series([0.001], index=pd.DatetimeIndex(["2024-01-03 08:00:00.004"]))}
    out = {}
    for name, wt in (("short", -0.5), ("long", 0.5)):
        res = simulate(w([wt, wt, wt]), "perp", {"X": perp_bars({})}, funding, fee=0.0, slippage_bps=0)
        out[name] = res
    assert out["long"].funding_paid == pytest.approx(0.001 * 50 * 100.0)  # pays 5 USDT
    assert out["short"].funding_paid == pytest.approx(-0.001 * 50 * 100.0)  # receives 5
    assert out["long"].equity_hourly.iloc[-1] == pytest.approx(10_000 - 5.0)
    assert out["short"].equity_hourly.iloc[-1] == pytest.approx(10_000 + 5.0)


def test_perp_equity_continuous_across_funding_without_position():
    funding = {"X": pd.Series([0.01, 0.01], index=pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 16:00"]))}
    res = simulate(w([0.0, 0.0]), "perp", {"X": perp_bars({})}, funding, fee=0.0005, slippage_bps=5)
    assert (res.equity_hourly == 10_000).all()
    assert res.funding_paid == 0.0


def test_perp_liquidation_on_adverse_wick_at_3x():
    # 3x long at 100: equity 10000, 300 contracts. Wick low 66 -> equity_adv = 10000-300*34 < 0.005*gross
    wick = {"2024-01-02 05:00": (101.0, 66.0)}
    res = simulate(w([3.0, 3.0]), "perp", {"X": perp_bars({}, wick=wick)}, {}, fee=0.0005, slippage_bps=0)
    assert len(res.liquidations) == 1
    ev = res.liquidations[0]
    assert ev["time"] == pd.Timestamp("2024-01-02 06:00")  # end of the wick bar
    assert res.equity_hourly.loc["2024-01-02 06:00"] < 1_000
    assert res.trades.iloc[-1]["kind"] == "liquidation"
    # nothing left open afterwards (next decision, same weights, re-enters but no phantom position)
    assert res.max_gross >= 2.9


def test_perp_no_liquidation_at_1x_with_same_wick():
    wick = {"2024-01-02 05:00": (101.0, 66.0)}
    res = simulate(w([1.0, 1.0]), "perp", {"X": perp_bars({}, wick=wick)}, {}, fee=0.0005, slippage_bps=0)
    assert res.liquidations == []


def test_band_suppresses_small_changes():
    params = TrendParams()

    def band(cur, tgt):
        return apply_band(cur, tgt, params)

    res = simulate(w([0.20, 0.21, 0.22]), "spot", {"X": spot_bars({})}, {}, fee=0.001, slippage_bps=5, band=band)
    assert len(res.trades) == 1  # only the entry; +0.01 changes are inside the band
    res2 = simulate(w([0.20, 0.21, 0.22]), "spot", {"X": spot_bars({})}, {}, fee=0.001, slippage_bps=5)
    assert len(res2.trades) == 3


def test_uncovered_coin_is_held_and_counted():
    bars_x = spot_bars({})
    bars_y = spot_bars({}).copy()
    bars_y.loc["2024-01-03 01:00", "close"] = np.nan  # no data at the second execution instant
    wd = pd.DataFrame({"X": [0.3, 0.3], "Y": [0.3, 0.0]}, index=pd.date_range(T0, periods=2, freq="D"))
    res = simulate(wd, "spot", {"X": bars_x, "Y": bars_y}, {}, fee=0.001, slippage_bps=5)
    assert res.coverage_warnings == 1
    assert res.exposure_daily["Y"].iloc[-1] > 0.25  # Y is still held


def test_deterministic():
    kw = dict(fee=0.0005, slippage_bps=5)
    funding = {"X": pd.Series([0.0001], index=pd.DatetimeIndex(["2024-01-02 08:00"]))}
    a = simulate(w([0.3, -0.2, 0.1]), "perp", {"X": perp_bars({"2024-01-03 00:00": 95.0})}, funding, **kw)
    b = simulate(w([0.3, -0.2, 0.1]), "perp", {"X": perp_bars({"2024-01-03 00:00": 95.0})}, funding, **kw)
    pd.testing.assert_series_equal(a.equity_hourly, b.equity_hourly)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert (a.fees_paid, a.funding_paid, a.turnover) == (b.fees_paid, b.funding_paid, b.turnover)
