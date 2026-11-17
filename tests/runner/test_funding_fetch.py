"""fetch_funding: live USD-M funding history. urlopen is mocked; no network."""
import io
import json
import urllib.error

import pandas as pd

from vv.runner import data as D


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _payload(rows):
    return json.dumps([{"symbol": "BTCUSDT", "fundingTime": t, "fundingRate": r} for t, r in rows]).encode()


def test_parses_rates_indexed_by_utc_funding_time(monkeypatch):
    seen = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return _Resp(_payload([(1704067200000, "0.0001"), (1704096000000, "-0.00025")]))

    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen)
    s = D.fetch_funding("BTCUSDT", 1704067200000)
    assert isinstance(s, pd.Series)
    assert list(s.index) == [pd.Timestamp("2024-01-01 00:00", tz="UTC"), pd.Timestamp("2024-01-01 08:00", tz="UTC")]
    assert s.iloc[0] == 0.0001 and s.iloc[1] == -0.00025
    assert seen[0].startswith("https://fapi.binance.com/fapi/v1/fundingRate?")
    assert "symbol=BTCUSDT" in seen[0] and "startTime=1704067200000" in seen[0] and "limit=1000" in seen[0]


def test_any_error_returns_empty_series(monkeypatch):
    def boom(url, timeout=None):
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(D.urllib.request, "urlopen", boom)
    s = D.fetch_funding("BTCUSDT", 0)
    assert s.empty and isinstance(s.index, pd.DatetimeIndex)


def test_malformed_payload_returns_empty_series(monkeypatch):
    monkeypatch.setattr(D.urllib.request, "urlopen", lambda url, timeout=None: _Resp(b"<html>451</html>"))
    assert D.fetch_funding("BTCUSDT", 0).empty


def test_funding_with_live_falls_back_to_live_fetch(tmp_path):
    import pandas as pd

    from vv.runner.engine import funding_with_live

    day = pd.Timestamp("2026-11-10", tz="UTC")
    prints = pd.Series([0.0001, 0.0002, -0.00005],
                       index=pd.to_datetime(["2026-11-10 00:00", "2026-11-10 08:00", "2026-11-10 16:00"], utc=True))
    calls = []

    def fake_fetch(symbol, since_ms):
        calls.append(symbol)
        return prints

    load = funding_with_live(tmp_path, fetch=fake_fetch)  # empty archive dir
    assert abs(load("BTCUSDT", day) - 0.00025) < 1e-12
    assert load("BTCUSDT", day + pd.Timedelta(days=1)) is None
    assert calls == ["BTCUSDT"]  # cached: one live call per symbol


def test_funding_with_live_empty_live_returns_none(tmp_path):
    import pandas as pd

    from vv.runner.engine import funding_with_live

    load = funding_with_live(tmp_path, fetch=lambda s, m: pd.Series(dtype=float))
    assert load("ETHUSDT", pd.Timestamp("2026-11-10", tz="UTC")) is None
