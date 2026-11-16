"""Exploratory: what happened last week, how, and what our signals said beforehand.

Fetches recent bars from data-api.binance.vision (public, no key) and extends the
stored daily history with them. Signals are computed from data up to the close of
the day BEFORE the week starts, so they are what we would actually have known.

Usage: .venv/bin/python -m research.last_week --start 2026-11-09 --end 2026-11-16
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://data-api.binance.vision/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]
COLS = ["open_time", "open", "high", "low", "close", "volume"]


def fetch(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, t = [], int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while t < end_ms:
        url = f"{API}?symbol={symbol}&interval={interval}&startTime={t}&endTime={end_ms - 1}&limit=1000"
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.load(r)
        if not batch:
            break
        rows += [b[:6] for b in batch]
        t = batch[-1][0] + 1
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=COLS)
    ts = df["open_time"].astype("int64")
    ts = ts.where(ts < 1e14, ts // 1000)  # tolerate microsecond timestamps
    df.index = pd.to_datetime(ts, unit="ms", utc=True)
    return df[COLS[1:]].astype(float)


def stored_daily(symbol: str, data_dir: Path) -> pd.Series:
    df = pd.read_parquet(data_dir / "klines" / f"{symbol}_1m.parquet", columns=["close"])
    s = df["close"].resample("1D").last().dropna()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def px(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:,.4g}"


def pct(x: float) -> str:
    return f"{x:+.1%}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-11-09")
    ap.add_argument("--end", default="2026-11-16")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ws, we = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    out = Path(a.out or f"docs/research/LAST_WEEK_{a.start}.md")
    d = Path(a.data_dir)

    L = [f"# Last week ({ws:%d %b} to {(we - pd.Timedelta(days=1)):%d %b %Y}): what happened and did we see it coming", "",
         "*Exploratory. Hourly and daily Binance spot bars (public API). Signals use only data up to the "
         f"close of {(ws - pd.Timedelta(days=1)):%d %b}, i.e. what we would have known before the week.*", ""]

    move_rows, sig_rows, path_rows, big_rows = [], [], [], []
    for sym in SYMBOLS:
        h = fetch(sym, "1h", ws - pd.Timedelta(days=35), we)
        hist = stored_daily(sym, d)
        recent_d = h["close"].resample("1D").last().dropna()
        daily = pd.concat([hist[hist.index < recent_d.index[0]], recent_d])
        daily = daily[daily.index < we]

        wk = h[(h.index >= ws) & (h.index < we)]
        before = h[h.index < ws]
        prev_close = before["close"].iloc[-1]
        ret = wk["close"].iloc[-1] / prev_close - 1
        hi, lo = wk["high"].max(), wk["low"].min()
        hr = wk["close"].pct_change().dropna()
        base_hr = before["close"].pct_change().dropna().iloc[-24 * 28:]
        vol_ratio = hr.std() / base_hr.std()
        vol_x = wk["volume"].mean() / before["volume"].iloc[-24 * 28:].mean()
        move_rows.append(f"| {sym[:-4]} | {px(prev_close)} | {px(wk['close'].iloc[-1])} | {pct(ret)} | "
                         f"{pct(lo / prev_close - 1)} / {pct(hi / prev_close - 1)} | {vol_ratio:.1f}x | {vol_x:.1f}x |")

        # Signals as of the close before the week
        known = daily[daily.index < ws]
        c = known.iloc[-1]
        sma50, sma200 = known.rolling(50).mean().iloc[-1], known.rolling(200).mean().iloc[-1]
        # 4h EMA200 needs ~33 days of 4h bars; the fetched hourly window covers 35 days
        four_h = h["close"][h.index < ws].resample("4h").last().dropna()
        ema4 = four_h.ewm(span=200, adjust=False).mean().iloc[-1] if len(four_h) >= 200 else float("nan")
        lead = c / known.iloc[-29] - 1
        r200 = c > sma200
        r50 = sma50 > sma200
        regime = c >= ema4 if ema4 == ema4 else None
        sig_rows.append(
            f"| {sym[:-4]} | {pct(lead)} | {pct(c / ema4 - 1)} | {'IN (above)' if r200 else 'OUT (below)'} ({c / sma200 - 1:+.0%} vs avg) | "
            f"{'IN' if r50 else 'OUT'} | {'buys allowed' if regime else ('buys blocked' if regime is not None else 'n/a')} | "
            f"{pct(ret if r200 else 0.0)} | {pct(ret if r50 else 0.0)} | {pct(ret)} |")

        if sym == "BTCUSDT":
            dd = wk.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            for t, r in dd.iterrows():
                path_rows.append(f"| {t:%a %d %b} | {r.open:,.0f} | {r.high:,.0f} | {r.low:,.0f} | {r.close:,.0f} | "
                                 f"{pct(r.close / r.open - 1)} | {r.volume:,.0f} |")
            top = hr.abs().sort_values(ascending=False).head(5).index
            for t in sorted(top):
                big_rows.append(f"| {t:%a %d %b %H:%M} UTC | {pct(hr[t])} | {wk.loc[t, 'volume'] / before['volume'].iloc[-24 * 28:].mean():.1f}x |")

    L += ["## What happened", "",
          "| Coin | Close before week | Close end of week | Week return | Low / high in week | Hourly volatility vs prior 4 weeks | Volume vs prior 4 weeks |",
          "|---|---|---|---|---|---|---|", *move_rows, ""]
    L += ["## How it happened: BTC day by day (UTC)", "",
          "| Day | Open | High | Low | Close | Day change | Volume (BTC) |", "|---|---|---|---|---|---|---|", *path_rows, ""]
    L += ["### BTC's 5 biggest hourly moves", "", "| Hour | Move | Volume vs normal hour |", "|---|---|---|", *big_rows, ""]
    L += ["## Did we see it coming? Signals as of the close before the week", "",
          "| Coin | Prior 4 weeks | Price vs 4h EMA200 | 200-day rule | 50/200 rule | Bot regime gate | Week with 200-day rule | Week with 50/200 rule | Week just holding |",
          "|---|---|---|---|---|---|---|---|---|", *sig_rows, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
