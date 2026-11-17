"""Daily-close store (append-only parquet with ingestion timestamps) and gap detection."""
from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

API = "https://data-api.binance.vision/api/v3/klines"
STALE_HOURS = 26.0
GAP_WINDOW_DAYS = 180
Fetcher = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]  # (symbol, start, end_excl) -> close/volume
_COLS = ["symbol", "date", "close", "volume", "ingested_at"]


def fetch_daily(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Closed 1d bars in [start, end) from data-api.binance.vision (public, no key)."""
    rows, t, end_ms = [], int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    while t < end_ms:
        url = f"{API}?symbol={symbol}&interval=1d&startTime={t}&endTime={end_ms - 1}&limit=1000"
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.load(r)
        if not batch:
            break
        rows += [b[:6] for b in batch]
        t = batch[-1][0] + 1
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    ts = df["t"].astype("int64")
    df.index = pd.to_datetime(ts.where(ts < 1e14, ts // 1000), unit="ms", utc=True).rename("date")
    return df[["close", "volume"]].astype(float)


FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding(symbol: str, since_ms: int) -> pd.Series:
    """USD-M perp funding rates from startTime=since_ms, indexed by funding time (UTC).

    Any error (network, HTTP, malformed payload) returns an empty Series; the caller
    treats missing funding as a flagged gap rather than a crash."""
    empty = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype="float64", name="funding_rate")
    try:
        rows, t = [], int(since_ms)
        while True:
            url = f"{FUNDING_API}?symbol={symbol}&startTime={t}&limit=1000"
            with urllib.request.urlopen(url, timeout=30) as r:
                batch = json.load(r)
            if not batch:
                break
            rows += [(int(b["fundingTime"]), float(b["fundingRate"])) for b in batch]
            if len(batch) < 1000:
                break
            t = int(batch[-1]["fundingTime"]) + 1
            time.sleep(0.1)
    except Exception:  # noqa: BLE001 - any failure means "funding unavailable"
        return empty
    if not rows:
        return empty
    ts = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    s = pd.Series([r[1] for r in rows], index=pd.DatetimeIndex(ts, name="fundingTime"), name="funding_rate")
    return s[~s.index.duplicated(keep="last")].sort_index()


def expected_last_bar(now: pd.Timestamp) -> pd.Timestamp:
    """The most recent fully closed daily bar (UTC midnight of yesterday)."""
    return now.tz_convert("UTC").normalize() - pd.Timedelta(days=1)


class DailyStore:
    """<data_dir>/runner/daily.parquet: rows are only ever added, never rewritten."""

    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / "runner" / "daily.parquet"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame({"symbol": [], "date": pd.DatetimeIndex([], tz="UTC"), "close": [],
                                 "volume": [], "ingested_at": pd.DatetimeIndex([], tz="UTC")})
        return pd.read_parquet(self.path)

    def append(self, new: pd.DataFrame) -> int:
        old = self.load()
        have = set(zip(old["symbol"], old["date"], strict=True))
        keep = [(s, d) not in have for s, d in zip(new["symbol"], new["date"], strict=True)]
        add = new[keep]
        if add.empty:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = pd.concat([old, add], ignore_index=True)[_COLS]
        tmp = self.path.with_suffix(".tmp")
        out.to_parquet(tmp)
        tmp.replace(self.path)
        return len(add)


def update_store(store: DailyStore, symbols: list[str], fetch: Fetcher, now: pd.Timestamp,
                 history_days: int) -> tuple[dict[str, int], list[str]]:
    """Fetch missing closed bars per symbol. Returns ({symbol: rows added}, errors)."""
    last = expected_last_bar(now)
    end = last + pd.Timedelta(days=1)
    cur = store.load()
    added: dict[str, int] = {}
    errors: list[str] = []
    for s in symbols:
        have = cur[cur["symbol"] == s]["date"]
        start = (have.max() + pd.Timedelta(days=1)) if len(have) else last - pd.Timedelta(days=history_days)
        if start > last:
            added[s] = 0
            continue
        try:
            df = fetch(s, start, end)
        except Exception as e:  # network trouble: gap detection decides whether this is fatal
            errors.append(f"{s}: fetch failed ({type(e).__name__})")
            added[s] = 0
            continue
        df = df[(df.index >= start) & (df.index <= last)]
        new = pd.DataFrame({"symbol": s, "date": df.index, "close": df["close"].to_numpy(float),
                            "volume": df["volume"].to_numpy(float), "ingested_at": now.tz_convert("UTC")})
        added[s] = store.append(new)
    return added, errors


def wide(store: DailyStore, symbols: list[str], now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume) frames: UTC daily index up to the last closed bar, one column per symbol."""
    df = store.load()
    df = df[df["date"] <= expected_last_bar(now)]
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    vol = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    return close.reindex(columns=symbols), vol.reindex(columns=symbols)


def detect_gaps(close: pd.DataFrame, now: pd.Timestamp) -> list[str]:
    """Missing / stale / holed bars. Any entry means: skip trading, alert, exit non-zero."""
    now = now.tz_convert("UTC")
    exp = expected_last_bar(now)
    out: list[str] = []
    for s in close.columns:
        col = close[s].dropna()
        if col.empty:
            out.append(f"{s}: no data")
            continue
        last = col.index.max()
        age_h = (now - (last + pd.Timedelta(days=1))).total_seconds() / 3600
        if last < exp:
            out.append(f"{s}: last bar {last.date()} but expected {exp.date()} (age {age_h:.1f}h)")
        elif age_h > STALE_HOURS:
            out.append(f"{s}: stale, last bar closed {age_h:.1f}h ago")
        win = col[col.index > exp - pd.Timedelta(days=GAP_WINDOW_DAYS)]
        if len(win):
            full = pd.date_range(win.index.min(), last, freq="D", tz="UTC")
            holes = full.difference(win.index)
            if len(holes):
                out.append(f"{s}: {len(holes)} missing bar(s) in history, first {holes[0].date()}")
    return out
