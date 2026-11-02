#!/usr/bin/env python3
"""
Download Binance spot 1m monthly klines from data.binance.vision.

Each monthly zip is verified against its .CHECKSUM (sha256), parsed, and cached as
data/klines/_parts/{SYMBOL}-{YYYY-MM}.parquet so reruns resume. The zip is deleted
once it parses. One parquet per symbol is written to {out}/{SYMBOL}_1m.parquet with
index open_time (UTC, datetime64[ns]).

Usage:
    python scripts/fetch_klines.py --symbols ETHUSDT BTCUSDT --start 2019-01 --end 2026-10
"""
import argparse
import hashlib
import io
import logging
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL = "1m"
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KEPT_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume"]
RETRIES = 3
BACKOFF_S = 2.0
# Epoch ms is ~1.7e12 in 2026; epoch us is ~1.7e15. Anything above 1e14 is microseconds.
US_THRESHOLD = 1e14

log = logging.getLogger("fetch_klines")


class ChecksumError(RuntimeError):
    pass


class DownloadError(RuntimeError):
    pass


def month_range(start: str, end: str) -> list:
    """Inclusive list of 'YYYY-MM' strings."""
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    months = []
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def zip_name(symbol: str, month: str) -> str:
    return f"{symbol}-{INTERVAL}-{month}.zip"


def zip_url(symbol: str, month: str) -> str:
    return f"{BASE_URL}/{symbol}/{INTERVAL}/{zip_name(symbol, month)}"


def checksum_url(symbol: str, month: str) -> str:
    return zip_url(symbol, month) + ".CHECKSUM"


def normalise_open_time(raw) -> np.ndarray:
    """Epoch timestamps (ms or us, decided per value) -> UTC datetime64[ns]."""
    values = np.array([int(v) for v in raw], dtype=np.int64)
    factor = np.where(values > US_THRESHOLD, 1_000, 1_000_000).astype(np.int64)
    return (values * factor).astype("datetime64[ns]")


def parse_checksum(text: str) -> str:
    """First token of a '<sha256>  <filename>' checksum file."""
    tokens = text.split()
    if not tokens or len(tokens[0]) != 64:
        raise ChecksumError(f"unreadable checksum file: {text[:80]!r}")
    return tokens[0].lower()


def verify_checksum(payload: bytes, checksum_text: str) -> None:
    expected = parse_checksum(checksum_text)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ChecksumError(f"sha256 mismatch: expected {expected}, got {actual}")


def parse_klines_csv(data: bytes) -> pd.DataFrame:
    """Headerless Binance kline CSV -> frame indexed by UTC open_time."""
    raw = pd.read_csv(io.BytesIO(data), header=None, names=COLUMNS, dtype=str)
    if not raw.empty and not raw["open_time"].iloc[0].isdigit():
        raw = raw.iloc[1:]  # some files carry a header row
    out = pd.DataFrame({"open_time": normalise_open_time(raw["open_time"])})
    for col in KEPT_FLOAT_COLUMNS:
        out[col] = raw[col].astype("float64").to_numpy()
    out["trades"] = raw["trades"].astype("int64").to_numpy()
    return out.set_index("open_time")


def read_zip_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"{zip_path}: expected one CSV, found {names}")
        return parse_klines_csv(zf.read(names[0]))


def count_gaps(index: pd.DatetimeIndex) -> int:
    """Missing 1-minute slots between the first and last timestamp."""
    idx = pd.DatetimeIndex(index).unique().sort_values()
    if len(idx) < 2:
        return 0
    expected = pd.date_range(idx[0], idx[-1], freq="min")
    return len(expected) - len(idx)


def _get(url: str, timeout: float = 120) -> bytes | None:
    """GET with retries and backoff. Returns None on 404."""
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
        except OSError as e:  # URLError, timeouts, resets
            last_err = e
        if attempt < RETRIES:
            delay = BACKOFF_S * 2 ** attempt
            log.warning("GET %s failed (%s), retrying in %.0fs", url, last_err, delay)
            time.sleep(delay)
    raise DownloadError(f"GET {url} failed after {RETRIES + 1} attempts: {last_err}")


def download_month(symbol: str, month: str, zip_path: Path) -> bool:
    """Fetch zip and checksum, verify, write zip. False if the month does not exist (404)."""
    for attempt in range(RETRIES + 1):
        checksum = _get(checksum_url(symbol, month))
        if checksum is None:
            return False
        payload = _get(zip_url(symbol, month))
        if payload is None:
            return False
        try:
            verify_checksum(payload, checksum.decode("utf-8", "replace"))
        except ChecksumError as e:
            if attempt == RETRIES:
                raise
            log.warning("%s %s: %s, re-downloading", symbol, month, e)
            time.sleep(BACKOFF_S * 2 ** attempt)
            continue
        zip_path.write_bytes(payload)
        return True
    raise DownloadError(f"{symbol} {month}: no verified download")


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.parent / (path.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def load_month(symbol: str, month: str, out_dir: Path):
    """Parsed frame for one month (cached), or None if Binance has no such month."""
    part = out_dir / "_parts" / f"{symbol}-{month}.parquet"
    if part.exists():
        return pd.read_parquet(part)
    zip_dir = out_dir / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / zip_name(symbol, month)
    if not download_month(symbol, month, zip_path):
        log.warning("SKIP %s %s: 404 on data.binance.vision", symbol, month)
        return None
    try:
        df = read_zip_csv(zip_path)
    finally:
        zip_path.unlink(missing_ok=True)
    _atomic_parquet(df, part)
    return df


def build_symbol(symbol: str, months: list, out_dir: Path) -> dict:
    frames, skipped = [], []
    for month in months:
        df = load_month(symbol, month, out_dir)
        if df is None:
            skipped.append(month)
        else:
            frames.append(df)
            log.info("%s %s: %d rows", symbol, month, len(df))
    if not frames:
        raise RuntimeError(f"{symbol}: no months downloaded")

    full = pd.concat(frames).sort_index(kind="stable")
    full = full[~full.index.duplicated(keep="last")]

    out_path = out_dir / f"{symbol}_1m.parquet"
    _atomic_parquet(full, out_path)
    return {
        "symbol": symbol,
        "rows": len(full),
        "first": full.index[0],
        "last": full.index[-1],
        "gaps": count_gaps(full.index),
        "size_mb": out_path.stat().st_size / 1e6,
        "skipped": skipped,
        "path": out_path,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT"])
    parser.add_argument("--start", default="2019-01", help="YYYY-MM, inclusive")
    parser.add_argument("--end", default="2026-10", help="YYYY-MM, inclusive")
    parser.add_argument("--out", default="data/klines")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out)
    (out_dir / "_parts").mkdir(parents=True, exist_ok=True)

    months = month_range(args.start, args.end)
    for symbol in args.symbols:
        s = build_symbol(symbol, months, out_dir)
        print(
            f"{s['symbol']}: rows={s['rows']} first={s['first']} last={s['last']} "
            f"gaps={s['gaps']} size={s['size_mb']:.1f}MB path={s['path']}"
        )
        print(f"{s['symbol']}: skipped months (404): {s['skipped'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
