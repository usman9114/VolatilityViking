"""Exploratory (NOT pre-registered): what did the market do, what would perfect
hindsight have earned, and what do simple ex-ante timing rules actually deliver?

Daily closes from the 1m klines. Rules decide on yesterday's close (no look-ahead),
switch fully between coin and USDT, and pay 0.1% per switch.

Usage: .venv/bin/python -m research.explore_timing --out docs/research/TIMING_EXPLORATION.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEE = 0.001
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]


def daily_close(symbol: str, data_dir: Path) -> pd.Series:
    df = pd.read_parquet(data_dir / "klines" / f"{symbol}_1m.parquet", columns=["close"])
    return df["close"].resample("1D").last().dropna()


def stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return {"total": total, "cagr": cagr, "mdd": mdd}


def run_rule(close: pd.Series, in_market: pd.Series) -> tuple[pd.Series, int]:
    """Equity of a coin/USDT switch. in_market is decided on yesterday's close."""
    pos = in_market.shift(1).fillna(False).astype(float)
    r = close.pct_change().fillna(0.0)
    switches = pos.diff().abs().fillna(pos.iloc[0])
    eq = (1 + pos * r - switches * FEE).cumprod()
    return eq, int(switches.sum())


def rules(close: pd.Series) -> dict[str, pd.Series]:
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    return {
        "Buy and hold": pd.Series(True, index=close.index),
        "Price above 200-day average": (close > sma200) & sma200.notna(),
        "50-day above 200-day average": (sma50 > sma200) & sma200.notna(),
    }


def fmt(x: float) -> str:
    """Total return; large gains also shown as a multiple of the start value."""
    return f"{x:+,.0%}" if x < 4 else f"{x:+,.0%} ({1 + x:,.0f}x money)"


def px(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:,.4g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="docs/research/TIMING_EXPLORATION.md")
    a = ap.parse_args(argv)
    d = Path(a.data_dir)
    closes = {s: daily_close(s, d) for s in SYMBOLS}

    L = ["# Market timing: hindsight vs rules you could actually follow", "",
         "*Exploratory, not pre-registered. Daily closes from Binance spot 1m data. "
         "Rules decide on yesterday's close and pay 0.1% per switch.*", ""]

    # 1. What happened in 2025-26
    L += ["## 1. What the market did since 2025", "",
          "| Coin | 2025 peak | Date | Low after peak | Date | Peak to low | 31 Oct 2026 vs peak |",
          "|---|---|---|---|---|---|---|"]
    for s, c in closes.items():
        w = c["2025-01-01":]
        pk_t = w.idxmax()
        after = w[pk_t:]
        lo_t = after.idxmin()
        L.append(f"| {s[:-4]} | {px(w[pk_t])} | {pk_t:%Y-%m-%d} | {px(after[lo_t])} | {lo_t:%Y-%m-%d} | "
                 f"{after[lo_t] / w[pk_t] - 1:+.0%} | {w.iloc[-1] / w[pk_t] - 1:+.0%} |")

    # 2. Perfect hindsight, BTC, since 2025
    c = closes["BTCUSDT"]["2025-01-01":]
    pk_t = c.idxmax()
    lo_t = c[pk_t:].idxmin()
    hold = c.iloc[-1] / c.iloc[0] - 1
    one_sell = (c[pk_t] / c.iloc[0]) * (c.iloc[-1] / c[lo_t]) * (1 - FEE) ** 3 - 1
    double = (c[pk_t] / c.iloc[0]) * (1 + 2 * (c.iloc[-1] / c[lo_t] - 1)) * (1 - FEE) ** 3 - 1
    r = c.pct_change().dropna()
    perfect_daily = (1 + r.clip(lower=0)).prod() - 1
    L += ["", "## 2. BTC since 1 Jan 2025: what perfect hindsight would have earned", "",
          "| Approach | Return |", "|---|---|",
          f"| Just hold | {hold:+.0%} |",
          f"| Sell at the exact top ({pk_t:%d %b %Y}), buy back at the exact bottom ({lo_t:%d %b %Y}) | {one_sell:+.0%} |",
          f"| Same, but buy back at the bottom with 2x leverage (\"double down\") | {double:+.0%} |",
          f"| Know every day in advance whether it goes up or down | {fmt(perfect_daily)} |",
          "", "These need to know the exact top and bottom in advance. Nobody can; the question is "
          "what rules that only see the past would have done.", ""]

    # 3. Ex-ante rules
    for label, start in [("2019 to Oct 2026", "2019-01-01"), ("2025 to Oct 2026", "2025-01-01")]:
        L += [f"## 3{'a' if start.startswith('2019') else 'b'}. Rules you could actually follow, {label}", "",
              "| Coin | Rule | Return | Per year | Worst drop | Switches |", "|---|---|---|---|---|---|"]
        for s, full in closes.items():
            rs = rules(full)  # indicators use full history, so warm-up before `start` is real past data
            for name, sig in rs.items():
                eq, n = run_rule(full[start:], sig[start:])
                st = stats(eq)
                L.append(f"| {s[:-4]} | {name} | {fmt(st['total'])} | {st['cagr']:+.0%} | {st['mdd']:.0%} | {n} |")
        L.append("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
