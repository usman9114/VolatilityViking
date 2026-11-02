"""
Tests for scripts/fetch_klines.py. No network access.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

import fetch_klines as fk  # noqa: E402


# 2019-01-01T00:00:00Z and 2025-01-01T00:00:00Z
MS_OPEN_TIME = 1546300800000
US_OPEN_TIME = 1735689600000000


class TestNormaliseOpenTime:
    def test_millisecond_row(self):
        ts = fk.normalise_open_time([MS_OPEN_TIME])
        assert ts.dtype == np.dtype("datetime64[ns]")
        assert ts[0] == np.datetime64("2019-01-01T00:00:00", "ns")

    def test_microsecond_row(self):
        ts = fk.normalise_open_time([US_OPEN_TIME])
        assert ts.dtype == np.dtype("datetime64[ns]")
        assert ts[0] == np.datetime64("2025-01-01T00:00:00", "ns")

    def test_mixed_units_and_string_input(self):
        ts = fk.normalise_open_time([str(MS_OPEN_TIME + 60_000), US_OPEN_TIME + 60_000_000])
        assert list(ts) == [
            np.datetime64("2019-01-01T00:01:00", "ns"),
            np.datetime64("2025-01-01T00:01:00", "ns"),
        ]


PAYLOAD = b"1546300800000,1.0,1.0,1.0,1.0,0.5\n"


def _checksum_text(payload: bytes, name: str = "ETHUSDT-1m-2019-01.zip") -> str:
    return f"{hashlib.sha256(payload).hexdigest()}  {name}\n"


class TestChecksum:
    def test_accepts_intact_payload(self):
        fk.verify_checksum(PAYLOAD, _checksum_text(PAYLOAD))

    def test_rejects_tampered_byte(self):
        tampered = bytes([PAYLOAD[0] ^ 0x01]) + PAYLOAD[1:]
        with pytest.raises(fk.ChecksumError):
            fk.verify_checksum(tampered, _checksum_text(PAYLOAD))


class TestHelpers:
    def test_month_range_is_inclusive_across_years(self):
        assert fk.month_range("2019-11", "2020-02") == [
            "2019-11", "2019-12", "2020-01", "2020-02",
        ]

    def test_month_range_empty_when_reversed(self):
        assert fk.month_range("2020-02", "2019-11") == []

    def test_zip_url(self):
        assert fk.zip_url("ETHUSDT", "2019-01") == (
            "https://data.binance.vision/data/spot/monthly/klines/"
            "ETHUSDT/1m/ETHUSDT-1m-2019-01.zip"
        )

    def test_count_gaps_counts_missing_minutes(self):
        idx = pd.to_datetime(
            ["2019-01-01 00:00", "2019-01-01 00:01", "2019-01-01 00:04"]
        ).astype("datetime64[ns]")
        assert fk.count_gaps(pd.DatetimeIndex(idx)) == 2

    def test_parse_csv_without_header_and_with_mixed_units(self):
        csv = (
            f"{MS_OPEN_TIME},1.0,2.0,0.5,1.5,10.0,0,0,7,0,0,0\n"
            f"{US_OPEN_TIME},2.0,3.0,1.0,2.5,20.0,0,0,9,0,0,0\n"
        ).encode()
        df = fk.parse_klines_csv(csv)
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "trades"]
        assert df.index.name == "open_time"
        assert df.index[0] == np.datetime64("2019-01-01T00:00:00", "ns")
        assert df.index[1] == np.datetime64("2025-01-01T00:00:00", "ns")
        assert df["volume"].dtype == np.float64
        assert df["trades"].dtype == np.int64
        assert df["trades"].tolist() == [7, 9]
