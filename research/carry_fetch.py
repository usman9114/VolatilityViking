"""Download USDS-M perpetual funding rates and 1h klines (data.binance.vision).

Reuses the HTTP, checksum and timestamp helpers from scripts/fetch_klines.py.
Writes data/futures/{SYM}_funding.parquet and data/futures/{SYM}_perp_1h.parquet,
both indexed by UTC timestamp. Monthly zips are cached as parquet parts so reruns
resume.

Usage:
    python -m research.carry_fetch --start 2020-01 --end 2026-10
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_klines as fk  # noqa: E402

BASE = "https://data.binance.vision/data/futures/um/monthly"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT",
           "BNBUSDT", "AVAXUSDT"]
OUT_DIR = Path("data/futures")
log = logging.getLogger("carry_fetch")


def parse_funding_csv(data: bytes) -> pd.DataFrame:
    """calc_time, funding_interval_hours, last_funding_rate -> frame indexed by UTC time."""
    raw = pd.read_csv(io.BytesIO(data), dtype=str)
    out = pd.DataFrame({
        "time": fk.normalise_open_time(raw["calc_time"]),
        "rate": raw["last_funding_rate"].astype("float64").to_numpy(),
        "interval_h": raw["funding_interval_hours"].astype("int64").to_numpy()
        if "funding_interval_hours" in raw else 8,
    })
    return out.set_index("time")


def _fetch_zip(url: str, tmp: Path) -> bool:
    checksum = fk._get(url + ".CHECKSUM")
    if checksum is None:
        return False
    for attempt in range(fk.RETRIES + 1):
        payload = fk._get(url)
        if payload is None:
            return False
        try:
            fk.verify_checksum(payload, checksum.decode("utf-8", "replace"))
        except fk.ChecksumError:
            if attempt == fk.RETRIES:
                raise
            continue
        tmp.write_bytes(payload)
        return True
    return False


def _month(kind: str, symbol: str, month: str, out_dir: Path):
    part = out_dir / "_parts" / f"{kind}-{symbol}-{month}.parquet"
    if part.exists():
        return pd.read_parquet(part)
    if kind == "funding":
        url = f"{BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    else:
        url = f"{BASE}/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    zpath = out_dir / "_parts" / f"{kind}-{symbol}-{month}.zip"
    if not _fetch_zip(url, zpath):
        return None
    try:
        with zipfile.ZipFile(zpath) as zf:
            data = zf.read(zf.namelist()[0])
    finally:
        zpath.unlink(missing_ok=True)
    df = parse_funding_csv(data) if kind == "funding" else fk.parse_klines_csv(data)
    tmp = part.with_name(part.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, part)
    return df


def build(kind: str, symbol: str, months: list, out_dir: Path) -> pd.DataFrame | None:
    frames = []
    for m in months:
        df = _month(kind, symbol, m, out_dir)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return None
    full = pd.concat(frames).sort_index(kind="stable")
    full = full[~full.index.duplicated(keep="last")]
    name = f"{symbol}_funding.parquet" if kind == "funding" else f"{symbol}_perp_1h.parquet"
    full.to_parquet(out_dir / name)
    return full


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out)
    (out_dir / "_parts").mkdir(parents=True, exist_ok=True)
    months = fk.month_range(args.start, args.end)
    for sym in args.symbols:
        for kind in ("funding", "klines"):
            df = build(kind, sym, months, out_dir)
            if df is None:
                print(f"{sym} {kind}: no data")
            else:
                print(f"{sym} {kind}: rows={len(df)} {df.index[0]} -> {df.index[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
