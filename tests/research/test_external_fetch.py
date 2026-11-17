"""Tests for research/external_fetch.py (WP-R1 external data). No network, no data/."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from research import external_fetch as ef

METRIC_COLS = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


def _metrics_raw(times, values):
    """Raw 5-minute metrics rows; every metric column carries the same value."""
    df = pd.DataFrame({"create_time": times})
    for col in METRIC_COLS:
        df[col] = values
    return df


def test_fred_dot_is_nan():
    csv = b"observation_date,DGS10\n2024-01-02,3.95\n2024-01-03,.\n2024-01-04,4.01\n"
    s = ef.parse_fred_csv(csv, "DGS10")
    assert s.dtype == np.float64
    assert list(s.isna()) == [False, True, False]
    assert s.iloc[0] == pytest.approx(3.95)
    assert s.iloc[2] == pytest.approx(4.01)


def test_metrics_day_takes_last_row():
    # Last row of 2024-03-01 by time is 23:55, even though it appears first in the file.
    raw = _metrics_raw(
        ["2024-03-01 23:55:00", "2024-03-01 00:00:00", "2024-03-01 12:00:00", "2024-03-02 00:00:00"],
        [3.0, 1.0, 2.0, 4.0],
    )
    out = ef.aggregate_metrics(raw)
    assert list(out.columns) == METRIC_COLS
    assert len(out) == 2
    assert out.loc["2024-03-01", "sum_open_interest"] == 3.0
    assert out.loc["2024-03-02", "sum_open_interest"] == 4.0


def test_epoch_normalisation_by_magnitude():
    assert ef.epoch_to_utc([1704067200000])[0] == pd.Timestamp("2024-01-01")  # ms
    assert ef.epoch_to_utc([1704067200])[0] == pd.Timestamp("2024-01-01")  # seconds
    assert ef.epoch_to_utc([1735689600000000])[0] == pd.Timestamp("2025-01-01")  # us
    assert ef.epoch_to_utc([1735693200000000])[0] == pd.Timestamp("2025-01-01 01:00")  # us, intraday


def test_output_index_is_daily_utc_named_date():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2024-01-03 08:00", "2024-01-01 16:00", "2024-01-01 00:00"]),
    )
    out = ef._finalise(df)
    assert out.index.name == "date"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique
    assert (out.index == out.index.normalize()).all()
    assert len(out) == 2


def test_premium_daily_mean_of_hourly_closes_mixed_units():
    # Two days of hourly rows: ms timestamps on day 1, us timestamps on day 2.
    rows = [
        [1704067200000, 0, 0, 0, 0.001, 0],
        [1704070800000, 0, 0, 0, 0.003, 0],
        [1704153600000000, 0, 0, 0, -0.001, 0],
        [1704157200000000, 0, 0, 0, 0.001, 0],
    ]
    raw = pd.DataFrame(rows)
    out = ef.premium_daily_mean(raw)
    assert out.index.name == "date"
    assert out.loc["2024-01-01", "perp_premium"] == pytest.approx(0.002)
    assert out.loc["2024-01-02", "perp_premium"] == pytest.approx(0.0)


def test_premium_zip_with_and_without_header(tmp_path):
    body = "1704067200000,0,0,0,0.001,0\n1704070800000,0,0,0,0.003,0\n"
    header = "open_time,open,high,low,close,volume\n"
    for name, text in [("nohdr", body), ("hdr", header + body)]:
        path = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{name}.csv", text)
        raw = ef.read_premium_zip(path)
        assert len(raw) == 2
        assert ef.premium_daily_mean(raw).loc["2024-01-01", "perp_premium"] == pytest.approx(0.002)


def test_funding_daily_sum_of_prints():
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00"],
        name="time",
    )
    funding = pd.DataFrame({"rate": [0.0001, 0.0002, -0.0001, 0.0005], "interval_h": 8}, index=idx)
    out = ef.funding_daily_sum(funding)
    assert out.index.name == "date"
    assert out.loc["2024-01-01", "funding_rate"] == pytest.approx(0.0002)
    assert out.loc["2024-01-02", "funding_rate"] == pytest.approx(0.0005)


def test_fear_greed_and_stablecoin_parsers():
    fng = b'{"data": [{"value": "12", "timestamp": "1704067200"}, {"value": "80", "timestamp": "1704153600"}]}'
    out = ef.parse_fear_greed(fng)
    assert out.loc["2024-01-01", "fear_greed"] == 12.0
    assert out.loc["2024-01-02", "fear_greed"] == 80.0

    stable = (
        b'[{"date": "1704067200", "totalCirculatingUSD": {"peggedUSD": 1000.0}},'
        b' {"date": "1704153600", "totalCirculatingUSD": {"peggedUSD": 1100.0}}]'
    )
    out = ef.parse_stablecoins(stable)
    assert out.loc["2024-01-02", "stablecoin_supply_usd"] == 1100.0
