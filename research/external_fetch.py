"""Fetch the external daily series named in research/feature_screen_prereg.json.

Writes data/external/*.parquet. Every file has a UTC daily DatetimeIndex named
"date" (naive timestamps at midnight UTC), sorted, without duplicates. No lagging
is applied here; the screen applies the publication lags.

Raw Binance zips are downloaded into a fresh directory under data/external/_raw/,
parsed with pandas, and deleted. 404s are skipped and recorded.

Usage:
    .venv/bin/python -m research.external_fetch
    .venv/bin/python -m research.external_fetch --incremental

--incremental fetches Binance daily zips only for days after each existing file's
last date (up to yesterday UTC) and merges them in. The all-history series (FRED,
Fear & Greed, stablecoins) are re-fetched in full, as before; funding is local.
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT_DIR = Path("data/external")
RAW_DIR = OUT_DIR / "_raw"
FUTURES_DIR = Path("data/futures")

WORKERS = 8
RETRIES = 3
TIMEOUT_S = 60
USER_AGENT = "research-external-fetch/1.0"

FRED_IDS = ["DTWEXBGS", "DGS10", "VIXCLS", "NASDAQCOM", "WALCL", "RRPONTSYD", "WTREGEN"]
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
STABLECOIN_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
BINANCE_BASE = "https://data.binance.vision/data/futures/um"

METRIC_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
FUNDING_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]
METRIC_START, METRIC_END = "2021-12-01", "2026-11-15"
PREMIUM_MONTHS = ("2020-01", "2026-10")
PREMIUM_DAILY = ("2026-11-01", "2026-11-15")

METRIC_COLS = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


class Missing(Exception):
    """HTTP 404: the resource does not exist. Skipped and recorded."""


class FetchError(Exception):
    """Still failing after all retries."""


@dataclass
class Ledger:
    missing: list[tuple[str, str]] = field(default_factory=list)  # (file, url)
    failed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- HTTP

def _get(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise Missing(url) from exc
            last = exc
        except OSError as exc:  # URLError and timeouts are OSError subclasses
            last = exc
        if attempt < RETRIES:
            time.sleep(2**attempt)
    raise FetchError(f"{url}: {last}")


def _run_many(
    fn: Callable[[Any], pd.DataFrame | None],
    items: list[Any],
    label: str,
    ledger: Ledger,
) -> list[pd.DataFrame]:
    """Run fn over items concurrently. Keeps input order; records 404s and failures."""
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [(item, pool.submit(fn, item)) for item in items]
        for item, fut in futures:
            try:
                result = fut.result()
            except Missing as exc:
                ledger.missing.append((label, str(exc)))
                continue
            except FetchError as exc:
                ledger.failed.append(f"{label}: {exc}")
                continue
            if result is not None and len(result):
                frames.append(result)
    return frames


def _load_zip(url: str, raw_dir: Path, reader: Callable[[Path], pd.DataFrame]) -> pd.DataFrame:
    """Download one zip into raw_dir, parse it with pandas, delete it."""
    dest = raw_dir / url.rsplit("/", 1)[-1]
    dest.write_bytes(_get(url))  # raises Missing on 404
    try:
        return reader(dest)
    finally:
        dest.unlink(missing_ok=True)


# ---------------------------------------------------------------- parsing helpers

def epoch_to_utc(values) -> pd.DatetimeIndex:
    """Epoch seconds, milliseconds or microseconds (chosen by magnitude) -> naive UTC."""
    arr = np.asarray(pd.to_numeric(pd.Series(list(values))), dtype="int64")
    factor = np.select([arr >= 10**14, arr >= 10**11], [1, 1000], default=10**6)
    return pd.DatetimeIndex(pd.to_datetime(arr * factor, unit="us"))


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Force a UTC daily index named 'date', sorted, unique (last value wins)."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.normalize().astype("datetime64[ns]")
    out = out.sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="last")]
    return out.rename_axis("date")


# ---------------------------------------------------------------- FRED / F&G / stablecoins

def parse_fred_csv(data: bytes, series_id: str) -> pd.Series:
    """fredgraph.csv -> float series; '.' (missing) becomes NaN."""
    raw = pd.read_csv(io.BytesIO(data), dtype=str)
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]))
    values = pd.to_numeric(raw.iloc[:, 1].replace(".", np.nan), errors="coerce")
    return pd.Series(values.to_numpy(dtype="float64"), index=dates, name=series_id)


def parse_fear_greed(payload: bytes) -> pd.DataFrame:
    rows = json.loads(payload)["data"]
    df = pd.DataFrame(
        {"fear_greed": [float(r["value"]) for r in rows]},
        index=epoch_to_utc([r["timestamp"] for r in rows]),
    )
    return _finalise(df)


def parse_stablecoins(payload: bytes) -> pd.DataFrame:
    rows = json.loads(payload)
    df = pd.DataFrame(
        {"stablecoin_supply_usd": [float(r["totalCirculatingUSD"]["peggedUSD"]) for r in rows]},
        index=epoch_to_utc([r["date"] for r in rows]),
    )
    return _finalise(df)


def _fetch_fred() -> pd.DataFrame:
    series = []
    for sid in FRED_IDS:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        series.append(parse_fred_csv(_get(url), sid))
    return _finalise(pd.concat(series, axis=1, sort=True))


# ---------------------------------------------------------------- Binance

def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """5-minute metrics rows -> one row per UTC day, taken from the day's last row."""
    df = raw.copy()
    df["create_time"] = pd.to_datetime(df["create_time"])
    df = df.sort_values("create_time", kind="stable")
    df["date"] = df["create_time"].dt.normalize()
    last = df.groupby("date", sort=True).tail(1).set_index("date")
    return _finalise(last[METRIC_COLS].astype("float64"))


def read_metrics_zip(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="zip")


def _metrics_url(sym: str, day: pd.Timestamp) -> str:
    return f"{BINANCE_BASE}/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip"


def _fetch_metrics(sym: str, raw_dir: Path, ledger: Ledger,
                   start: str = METRIC_START, end: str = METRIC_END) -> pd.DataFrame | None:
    days = pd.date_range(start, end, freq="D")
    fn = functools.partial(lambda d: _load_zip(_metrics_url(sym, d), raw_dir, read_metrics_zip))
    frames = _run_many(fn, list(days), f"binance_metrics_{sym}.parquet", ledger)
    if not frames:
        return None
    return aggregate_metrics(pd.concat(frames, ignore_index=True))


def read_premium_zip(path: Path) -> pd.DataFrame:
    """Premium klines zip; monthly files have no header, daily files do."""
    raw = pd.read_csv(path, compression="zip", header=None, dtype=str)
    if pd.isna(pd.to_numeric(raw.iloc[0, 0], errors="coerce")):
        raw = raw.iloc[1:]
    return raw.reset_index(drop=True)


def premium_daily_mean(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly premium klines (open_time, open, high, low, close, ...) -> daily mean of close."""
    ts = epoch_to_utc(raw.iloc[:, 0])
    close = pd.to_numeric(raw.iloc[:, 4]).to_numpy(dtype="float64")
    hourly = pd.Series(close, index=ts)
    daily = hourly.groupby(hourly.index.normalize()).mean()
    return _finalise(daily.to_frame("perp_premium"))


def _premium_urls(sym: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> list[str]:
    """Full history (monthly + daily) by default; with start/end, daily zips only."""
    if start is not None and end is not None:
        days = pd.date_range(start, end, freq="D")
        return [f"{BINANCE_BASE}/daily/premiumIndexKlines/{sym}/1h/{sym}-1h-{d:%Y-%m-%d}.zip" for d in days]
    months = pd.period_range(PREMIUM_MONTHS[0], PREMIUM_MONTHS[1], freq="M")
    monthly = [
        f"{BINANCE_BASE}/monthly/premiumIndexKlines/{sym}/1h/{sym}-1h-{m.strftime('%Y-%m')}.zip"
        for m in months
    ]
    days = pd.date_range(PREMIUM_DAILY[0], PREMIUM_DAILY[1], freq="D")
    daily = [
        f"{BINANCE_BASE}/daily/premiumIndexKlines/{sym}/1h/{sym}-1h-{d:%Y-%m-%d}.zip" for d in days
    ]
    return monthly + daily


def _fetch_premium(sym: str, raw_dir: Path, ledger: Ledger, urls: list[str] | None = None) -> pd.DataFrame | None:
    fn = functools.partial(lambda url: _load_zip(url, raw_dir, read_premium_zip))
    frames = _run_many(fn, urls if urls is not None else _premium_urls(sym),
                       f"binance_premium_{sym}.parquet", ledger)
    if not frames:
        return None
    return premium_daily_mean(pd.concat(frames, ignore_index=True))


def funding_daily_sum(funding: pd.DataFrame) -> pd.DataFrame:
    """Funding prints (index = settlement time, column 'rate') -> daily sum."""
    idx = funding.index
    if not pd.api.types.is_datetime64_any_dtype(idx):
        idx = epoch_to_utc(idx)
    s = pd.Series(funding["rate"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(idx))
    daily = s.groupby(s.index.normalize()).sum()
    return _finalise(daily.to_frame("funding_rate"))


def _fetch_funding(sym: str) -> pd.DataFrame:
    # Reuses the carry study's download (research/carry_fetch.py writes these files).
    path = FUTURES_DIR / f"{sym}_funding.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run research.carry_fetch first")
    return funding_daily_sum(pd.read_parquet(path))


# ---------------------------------------------------------------- incremental

def incremental_start(name: str, out_dir: Path, default_start: pd.Timestamp) -> pd.Timestamp:
    """Day after the file's last date, or default_start when the file does not exist yet."""
    path = out_dir / name
    if not path.exists():
        return pd.Timestamp(default_start)
    last = pd.read_parquet(path).index.max()
    return pd.Timestamp(last).normalize() + pd.Timedelta(days=1)


def merge_existing(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Old history plus new rows; for a day present in both, the new row wins."""
    return _finalise(pd.concat([old, new]))


# ---------------------------------------------------------------- orchestration

def build(raw_root: Path = RAW_DIR, out_dir: Path = OUT_DIR, incremental: bool = False,
          today: pd.Timestamp | None = None) -> tuple[dict[str, pd.DataFrame], Ledger]:
    ledger = Ledger()
    results: dict[str, pd.DataFrame] = {}
    yesterday = None
    if incremental:
        now = pd.Timestamp.now(tz="UTC") if today is None else pd.Timestamp(today)
        yesterday = now.tz_localize(None).normalize() - pd.Timedelta(days=1)

    def guard(name: str, fn: Callable[[], pd.DataFrame | None]) -> None:
        try:
            df = fn()
        except Exception as exc:  # recorded and reported; other files still build
            ledger.failed.append(f"{name}: {exc}")
            return
        if df is None or df.empty:
            ledger.failed.append(f"{name}: no data")
        else:
            results[name] = df

    raw_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=raw_root))
    try:
        guard("fred.parquet", _fetch_fred)
        guard("fear_greed.parquet", lambda: parse_fear_greed(_get(FNG_URL)))
        guard("stablecoins.parquet", lambda: parse_stablecoins(_get(STABLECOIN_URL)))
        for sym in METRIC_SYMBOLS:
            name = f"binance_metrics_{sym}.parquet"
            if yesterday is None:
                guard(name, lambda s=sym: _fetch_metrics(s, run_dir, ledger))
                continue
            start = incremental_start(name, out_dir, pd.Timestamp(METRIC_START))
            guard(name, lambda s=sym, st=start: _merged(
                name, out_dir, lambda: _fetch_metrics(s, run_dir, ledger, st.strftime("%Y-%m-%d"),
                                                      yesterday.strftime("%Y-%m-%d"))))
        for sym in METRIC_SYMBOLS:
            name = f"binance_premium_{sym}.parquet"
            if yesterday is None:
                guard(name, lambda s=sym: _fetch_premium(s, run_dir, ledger))
                continue
            start = incremental_start(name, out_dir, pd.Timestamp(PREMIUM_DAILY[0]))
            urls = _premium_urls(sym, start, yesterday) if start <= yesterday else []
            guard(name, lambda s=sym, u=urls: _merged(
                name, out_dir, lambda: _fetch_premium(s, run_dir, ledger, u)))
        for sym in FUNDING_SYMBOLS:
            guard(f"funding_{sym}.parquet", lambda s=sym: _fetch_funding(s))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return results, ledger


def _merged(name: str, out_dir: Path, fetch: Callable[[], pd.DataFrame | None]) -> pd.DataFrame | None:
    """Incremental: new rows merged onto the existing file. Nothing new keeps the file as it is."""
    new = fetch()
    path = out_dir / name
    if new is None or new.empty:
        return pd.read_parquet(path) if path.exists() else None
    return merge_existing(new, pd.read_parquet(path)) if path.exists() else new


def write(results: dict[str, pd.DataFrame], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        tmp = out_dir / (name + ".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, out_dir / name)


def summary_table(results: dict[str, pd.DataFrame]) -> str:
    lines = [f"{'file':<30} {'rows':>6}  {'first':<10}  {'last':<10}  {'NaN':>6}"]
    for name, df in results.items():
        nan = float(df.isna().to_numpy().mean())
        lines.append(
            f"{name:<30} {len(df):>6}  {df.index.min():%Y-%m-%d}  {df.index.max():%Y-%m-%d}  {nan:6.1%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--incremental", action="store_true",
                    help="fetch only Binance days after each file's last date")
    args = ap.parse_args(argv)

    results, ledger = build(out_dir=Path(args.out), incremental=args.incremental)
    write(results, Path(args.out))
    print(summary_table(results))
    if ledger.missing:
        print("\n404s skipped (file: count):")
        for name, count in sorted(Counter(f for f, _ in ledger.missing).items()):
            print(f"  {name}: {count}")
    for msg in ledger.failed:
        print(f"FAILED {msg}", file=sys.stderr)
    return 1 if ledger.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
