"""G0 edge pre-check: does ETH/BTC mean-revert at grid horizons?

A long-only spot grid is short volatility / long mean reversion. On a random
walk it has zero edge before fees. This script measures, on 1m Binance klines:

1. Lo-MacKinlay variance ratios (heteroskedasticity-robust z*, stationary
   block-bootstrap CI), pooled and per calendar year.
2. DFA Hurst exponent on 5m returns over 1h-1d scales.
3. Typical 1h / 4h bar range versus the maker round-trip fee.
4. A literal PASS/FAIL verdict under the G0 rule.

Usage:
    python -m research.g0_variance_ratio --symbols ETHUSDT BTCUSDT \
        --out docs/research/G0_RESULTS.md
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260101
REPS = 500
BARS_PER_DAY_5M = 288
BARS_PER_DAY_1M = 1440

# Horizons for the 5m base (name, q in 5m bars).
HORIZONS_5M = [("15m", 3), ("30m", 6), ("1h", 12), ("2h", 24),
               ("4h", 48), ("8h", 96), ("1d", 288)]
YEARLY_HORIZONS = ["1h", "4h"]
CONTAMINATED = [("5m (1m base)", 5)]

MAKER_ROUND_TRIP = 0.0015            # 0.15%, BNB-discounted maker, both legs
RANGE_THRESHOLD = 2.5 * MAKER_ROUND_TRIP   # 0.375%
MIN_YEAR_DAYS = 30                   # a calendar year needs this much data to count


# --------------------------------------------------------------------------
# Variance ratio
# --------------------------------------------------------------------------
@dataclass
class VRResult:
    q: int
    vr: float
    z: float   # heteroskedasticity-robust z*
    p: float   # two-sided normal p-value for z*


def _vr_point(r: np.ndarray, qs) -> np.ndarray:
    """Lo-MacKinlay (1988) unbiased VR(q) with overlapping q-sums, for each q."""
    r = np.asarray(r, dtype=float)
    n = r.size
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    csum = np.concatenate(([0.0], np.cumsum(r)))
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        rq = csum[q:] - csum[:-q]
        m = q * (n - q + 1) * (1.0 - q / n)
        out[i] = np.sum((rq - q * mu) ** 2) / m / var1
    return out


def variance_ratio_table(r, qs) -> list[VRResult]:
    """VR(q), robust z* and p-value for each q."""
    r = np.asarray(r, dtype=float)
    qmax = max(qs)
    e2 = (r - r.mean()) ** 2
    denom = e2.sum() ** 2
    delta = np.zeros(qmax)  # delta[j] for j >= 1
    for j in range(1, qmax):
        delta[j] = np.dot(e2[j:], e2[:-j]) / denom
    vrs = _vr_point(r, qs)
    res = []
    for q, vr in zip(qs, vrs):
        j = np.arange(1, q)
        theta = np.sum((2.0 * (q - j) / q) ** 2 * delta[1:q])
        z = (vr - 1.0) / math.sqrt(theta) if theta > 0 else float("nan")
        p = 2.0 * stats.norm.sf(abs(z)) if np.isfinite(z) else float("nan")
        res.append(VRResult(q, float(vr), float(z), float(p)))
    return res


def variance_ratio(r, q: int) -> VRResult:
    return variance_ratio_table(r, [q])[0]


def stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric block lengths, wrap-around)."""
    t = np.arange(n)
    new = rng.random(n) < 1.0 / mean_block
    new[0] = True
    starts = rng.integers(0, n, size=int(new.sum()))
    block_id = np.cumsum(new) - 1
    last_start_t = np.maximum.accumulate(np.where(new, t, 0))
    return (starts[block_id] + t - last_start_t) % n


def vr_bootstrap_ci(r, qs, mean_block: float, reps: int = REPS,
                    seed: int = SEED, alpha: float = 0.05) -> np.ndarray:
    """Percentile CI of VR(q) under a stationary block bootstrap. Shape (len(qs), 2)."""
    r = np.asarray(r, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, len(qs)))
    for b in range(reps):
        idx = stationary_bootstrap_indices(r.size, mean_block, rng)
        draws[b] = _vr_point(r[idx], qs)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    return np.column_stack([lo, hi])


def vr_rows(r, horizons, mean_block, reps, seed) -> list[dict]:
    qs = [q for _, q in horizons]
    tab = variance_ratio_table(r, qs)
    ci = vr_bootstrap_ci(r, qs, mean_block, reps, seed)
    rows = []
    for (name, q), res, (lo, hi) in zip(horizons, tab, ci):
        rows.append(dict(horizon=name, q=q, vr=res.vr, z=res.z, p=res.p,
                         lo=float(lo), hi=float(hi), n=len(r),
                         sig_below=bool(hi < 1.0)))
    return rows


# --------------------------------------------------------------------------
# DFA
# --------------------------------------------------------------------------
def dfa_scales(smin: int, smax: int, num: int = 10) -> np.ndarray:
    return np.unique(np.round(np.geomspace(smin, smax, num)).astype(int))


def dfa_hurst(r, scales) -> float:
    """DFA-1 scaling exponent of the integrated, demeaned series (0.5 = no memory)."""
    r = np.asarray(r, dtype=float)
    y = np.cumsum(r - r.mean())
    logf = []
    for s in scales:
        k = y.size // s
        seg = y[: k * s].reshape(k, s)
        x = np.arange(s) - (s - 1) / 2.0
        slope = seg @ x / np.sum(x ** 2)
        resid = seg - seg.mean(axis=1, keepdims=True) - slope[:, None] * x[None, :]
        logf.append(0.5 * math.log(np.mean(resid ** 2)))
    return float(np.polyfit(np.log(scales), logf, 1)[0])


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_klines(symbol: str, data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "klines" / f"{symbol}_1m.parquet"
    df = pd.read_parquet(path, columns=["open", "high", "low", "close"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def log_returns(close: pd.Series, rule: str) -> tuple[pd.Series, int]:
    """Log returns of `close` resampled to `rule`. Returns (returns, empty bins forward-filled)."""
    c = close.resample(rule).last()
    gaps = int(c.isna().sum())
    c = c.ffill().dropna()
    return np.log(c).diff().dropna(), gaps


def range_stats(df: pd.DataFrame, rule: str) -> dict:
    bars = df.resample(rule).agg({"open": "first", "high": "max", "low": "min"}).dropna()
    rng = (bars["high"] - bars["low"]) / bars["open"]
    return dict(median=float(rng.median()), p25=float(rng.quantile(0.25)), n=len(rng))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyse_symbol(symbol: str, df: pd.DataFrame, reps: int, seed: int) -> dict:
    r5, gaps5 = log_returns(df["close"], "5min")
    r1, _ = log_returns(df["close"], "1min")
    out = dict(symbol=symbol, start=df.index[0], end=df.index[-1], gaps5=gaps5,
               n5=len(r5), nbars=len(df))
    out["vr_pooled"] = vr_rows(r5.values, HORIZONS_5M, BARS_PER_DAY_5M, reps, seed)
    out["vr_1m"] = vr_rows(r1.values, CONTAMINATED, BARS_PER_DAY_1M, reps, seed)

    yearly_h = [(n, q) for n, q in HORIZONS_5M if n in YEARLY_HORIZONS]
    out["vr_year"] = {}
    out["hurst_year"] = {}
    for year, ry in r5.groupby(r5.index.year):
        days = len(ry) / BARS_PER_DAY_5M
        if days < MIN_YEAR_DAYS:
            continue
        out["vr_year"][int(year)] = vr_rows(ry.values, yearly_h, BARS_PER_DAY_5M, reps, seed)
        out["hurst_year"][int(year)] = dfa_hurst(ry.values, dfa_scales(12, 288))
    out["hurst_pooled"] = dfa_hurst(r5.values, dfa_scales(12, 288))
    out["range"] = {"1h": range_stats(df, "1h"), "4h": range_stats(df, "4h")}
    return out


RECENT_FROM = 2024


def evaluate_g0(results: list[dict]) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Apply the G0 rule literally. Returns (passed, [(check, ok, detail)])."""
    checks = []
    for res in results:
        sym = res["symbol"]
        pooled = {row["horizon"]: row for row in res["vr_pooled"]}
        for h in YEARLY_HORIZONS:
            row = pooled[h]
            checks.append((f"{sym} pooled {h}: VR<1 and CI excludes 1", row["sig_below"],
                           f"VR={row['vr']:.3f}, CI [{row['lo']:.3f}, {row['hi']:.3f}]"))
            years = res["vr_year"]
            good = [y for y, rows in years.items()
                    for rw in rows if rw["horizon"] == h and rw["sig_below"]]
            ok = len(good) * 2 > len(years)
            checks.append((f"{sym} {h}: VR<1 with CI excluding 1 in a majority of years", ok,
                           f"{len(good)} of {len(years)} years"))
        med = res["range"]["4h"]["median"]
        checks.append((f"{sym} median 4h range > {RANGE_THRESHOLD:.3%}", med > RANGE_THRESHOLD,
                       f"median {med:.3%}"))
    return all(ok for _, ok, _ in checks), checks


def symbol_verdicts(results: list[dict]) -> list[dict]:
    """Per-symbol G0 verdict: every check for that symbol must hold.

    Also reports how many of the most recent years (>= RECENT_FROM) show
    significant mean reversion at 1h, as information (not part of the rule).
    """
    _, checks = evaluate_g0(results)
    out = []
    for res in results:
        sym = res["symbol"]
        mine = [(c, ok, d) for c, ok, d in checks if c.startswith(sym + " ")]
        recent = {y: rows for y, rows in res["vr_year"].items() if int(y) >= RECENT_FROM}
        recent_ok = sum(1 for rows in recent.values()
                        for rw in rows if rw["horizon"] == "1h" and rw["sig_below"])
        out.append(dict(symbol=sym, passed=all(ok for _, ok, _ in mine),
                        failed=[c[len(sym) + 1:] for c, ok, _ in mine if not ok],
                        details={c[len(sym) + 1:]: d for c, _, d in mine},
                        recent=f"{recent_ok} of {len(recent)}"))
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _vr_table(rows: list[dict]) -> list[str]:
    lines = ["| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        excl = "yes (<1)" if r["hi"] < 1 else ("yes (>1)" if r["lo"] > 1 else "no")
        lines.append(f"| {r['horizon']} | {r['q']} | {r['vr']:.3f} | {r['z']:.2f} | "
                     f"{r['p']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] | {excl} |")
    return lines


def render_report(results: list[dict], reps: int, seed: int) -> str:
    passed, checks = evaluate_g0(results)
    L = ["# G0 results: which coins mean-revert at grid horizons?", ""]
    for res in results:
        L.append(f"- {res['symbol']}: {res['start']:%Y-%m-%d} to {res['end']:%Y-%m-%d}, "
                 f"{res['nbars']:,} 1m bars ({res['gaps5']} empty 5m bins forward-filled).")
    L += ["", f"Method: Lo-MacKinlay VR on log returns, robust z*, stationary block bootstrap "
          f"(mean block 1 day, {reps} reps, seed {seed}), percentile 95% CI. "
          "Year rows need at least "
          f"{MIN_YEAR_DAYS} days of data.", ""]

    L += ["## Verdict per coin", "",
          "Each coin is gated on its own. Only coins that pass go on to the G1 replay.", "",
          f"| Coin | G0 | Pooled 1h | Pooled 4h | Years 1h | Years 4h | 1h since {RECENT_FROM} | Median 4h range |",
          "|---|---|---|---|---|---|---|---|"]
    for v in symbol_verdicts(results):
        d = v["details"]
        get = lambda key: next((val for k, val in d.items() if k.startswith(key)), "")
        L.append(f"| {v['symbol']} | {'**PASS**' if v['passed'] else 'fail'} | "
                 f"{get('pooled 1h')} | {get('pooled 4h')} | {get('1h:')} | {get('4h:')} | "
                 f"{v['recent']} | {get('median 4h')} |")
    L += ["", f"With {len(results)} coins tested, expect occasional false passes from chance; "
          "the majority-of-years rule limits this, and G1 (replay with fees) is the real test.", ""]

    L += ["## Overall (all coins together)", ""]
    if passed:
        L.append("**PASS.** Every G0 check below holds.")
    else:
        failed = [c for c, ok, _ in checks if not ok]
        L.append(f"**FAIL.** {len(failed)} of {len(checks)} checks fail, e.g. {failed[0]}.")
    L += ["", "| Check | Result | Detail |", "|---|---|---|"]
    for c, ok, d in checks:
        L.append(f"| {c} | {'pass' if ok else 'fail'} | {d} |")
    L += ["", "Rule applied literally: PASS only if VR<1 with the CI excluding 1 at 1h and 4h "
          "for every symbol, in a majority of years, and the median 4h range exceeds "
          f"{RANGE_THRESHOLD:.3%}.", ""]

    for res in results:
        L += [f"## {res['symbol']}: variance ratio, pooled (5m base)", ""]
        L += _vr_table(res["vr_pooled"])
        L += ["", "1m base, labelled contaminated by bid-ask bounce (kline closes are trade "
              "prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):", ""]
        L += _vr_table(res["vr_1m"])
        L.append("")

    L += ["## Stability: per calendar year (VR, 95% CI)", "",
          "| Symbol | Year | 1h VR | 1h CI | 4h VR | 4h CI | DFA Hurst |", "|---|---|---|---|---|---|---|"]
    for res in results:
        for year, rows in res["vr_year"].items():
            d = {r["horizon"]: r for r in rows}
            a, b = d["1h"], d["4h"]
            L.append(f"| {res['symbol']} | {year} | {a['vr']:.3f} | [{a['lo']:.3f}, {a['hi']:.3f}] | "
                     f"{b['vr']:.3f} | [{b['lo']:.3f}, {b['hi']:.3f}] | {res['hurst_year'][year]:.3f} |")
    L += ["", "## DFA Hurst exponent (5m returns, 1h-1d scales)", "",
          "| Symbol | Pooled H |", "|---|---|"]
    for res in results:
        L.append(f"| {res['symbol']} | {res['hurst_pooled']:.3f} |")
    L += ["", "H < 0.5 means mean reversion, 0.5 a random walk, > 0.5 trending.", ""]

    L += ["## Range vs fees: (high - low) / open", "",
          f"Maker round trip {MAKER_ROUND_TRIP:.2%} (BNB); 2.5x threshold {RANGE_THRESHOLD:.3%}.", "",
          "| Symbol | Bar | Median | 25th pct | Median vs 0.375% |", "|---|---|---|---|---|"]
    for res in results:
        for bar in ("1h", "4h"):
            s = res["range"][bar]
            L.append(f"| {res['symbol']} | {bar} | {s['median']:.3%} | {s['p25']:.3%} | "
                     f"{'above' if s['median'] > RANGE_THRESHOLD else 'below'} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT"])
    ap.add_argument("--out", default="docs/research/G0_RESULTS.md")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    results = []
    for sym in args.symbols:
        print(f"analysing {sym} ...", flush=True)
        results.append(analyse_symbol(sym, load_klines(sym, Path(args.data_dir)),
                                      args.reps, args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(results, args.reps, args.seed))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
