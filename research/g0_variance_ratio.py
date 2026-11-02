"""G0 edge pre-check: does ETH/BTC mean-revert at grid horizons?

A long-only spot grid is short volatility / long mean reversion. On a random
walk it has zero edge before fees. This script measures, on 1m Binance klines:

1. Lo-MacKinlay variance ratios (heteroskedasticity-robust z*, stationary
   block-bootstrap CI), pooled and per calendar year.
2. DFA Hurst exponent on 5m returns over 1h-1d scales.
3. Typical 1h / 4h bar range versus the maker round-trip fee.
4. A literal PASS/FAIL verdict under the G0 rule.

Each coin is evaluated twice in the same run: a baseline, and a robust variant
(5m returns winsorised at +-5 robust sigma per calendar month, listed event
windows excluded, Holm-adjusted pooled p-values). The first
`--exclude-listing-days` days of each symbol are dropped in both. The final G1
universe is the coins that pass both.

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

LISTING_DAYS = 30                    # drop this much data after each symbol's first bar
WINSOR_K = 5.0                       # winsorise at +-K robust sigmas
MAD_SCALE = 1.4826                   # MAD -> sigma for a normal
ALPHA = 0.05
DEFAULT_WINDOWS = [("2025-10-10 00:00", "2025-10-12 00:00")]   # Oct-2025 cascade


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
    """Re-centred bootstrap CI of VR(q) under a stationary block bootstrap. Shape (len(qs), 2).

    Plain percentile intervals drift toward 1 on short, heavy-tailed samples (the point
    estimate can fall outside), so the interval is shifted by the bootstrap bias:
    CI = VR_hat + (q_lo - median(draws), q_hi - median(draws)).
    """
    r = np.asarray(r, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, len(qs)))
    for b in range(reps):
        idx = stationary_bootstrap_indices(r.size, mean_block, rng)
        draws[b] = _vr_point(r[idx], qs)
    lo, hi, med = np.quantile(draws, [alpha / 2, 1 - alpha / 2, 0.5], axis=0)
    point = _vr_point(r, qs)
    return np.column_stack([point + (lo - med), point + (hi - med)])


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


def holm_adjust(p) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, in the input order (NaN counts as 1)."""
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out


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


def drop_listing_days(df: pd.DataFrame, days: float) -> pd.DataFrame:
    """Drop the first `days` days of data (post-listing price discovery)."""
    if days <= 0 or df.empty:
        return df
    return df[df.index >= df.index[0] + pd.Timedelta(days=days)]


def parse_windows(raw) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[(start, end), ...] strings -> UTC timestamp pairs, half-open [start, end)."""
    out = []
    for a, b in raw:
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end <= start:
            raise ValueError(f"exclude window end must be after start: {a} {b}")
        out.append((start, end))
    return out


def exclude_windows(obj, windows):
    """Drop rows with index in any half-open window [start, end)."""
    keep = np.ones(len(obj), dtype=bool)
    for start, end in windows:
        keep &= ~((obj.index >= start) & (obj.index < end))
    return obj[keep]


def winsorise_monthly(r: pd.Series, k: float = WINSOR_K) -> tuple[pd.Series, int]:
    """Clip returns at +-k robust sigmas, sigma = 1.4826 * MAD per calendar month.

    Returns (clipped copy, number of bars changed). A month with zero MAD is left alone.
    """
    keys = [r.index.year, r.index.month]
    med = r.groupby(keys).transform("median")
    sigma = MAD_SCALE * (r - med).abs().groupby(keys).transform("median")
    lim = (k * sigma).where(sigma > 0, np.inf)
    out = r.clip(lower=-lim, upper=lim)
    return out, int((out != r).sum())


def range_stats(df: pd.DataFrame, rule: str) -> dict:
    bars = df.resample(rule).agg({"open": "first", "high": "max", "low": "min"}).dropna()
    rng = (bars["high"] - bars["low"]) / bars["open"]
    return dict(median=float(rng.median()), p25=float(rng.quantile(0.25)), n=len(rng))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _vr_blocks(r5: pd.Series, reps: int, seed: int) -> tuple[list[dict], dict]:
    """Pooled VR rows and per-calendar-year VR rows (1h, 4h) for a 5m return series."""
    yearly_h = [(n, q) for n, q in HORIZONS_5M if n in YEARLY_HORIZONS]
    pooled = vr_rows(r5.values, HORIZONS_5M, BARS_PER_DAY_5M, reps, seed)
    years = {}
    for year, ry in r5.groupby(r5.index.year):
        if len(ry) / BARS_PER_DAY_5M < MIN_YEAR_DAYS:
            continue
        years[int(year)] = vr_rows(ry.values, yearly_h, BARS_PER_DAY_5M, reps, seed)
    return pooled, years


def analyse_symbol(symbol: str, df: pd.DataFrame, reps: int, seed: int,
                   listing_days: float = LISTING_DAYS, windows=None) -> dict:
    """Baseline and robust analysis of one symbol (listing days dropped from both)."""
    windows = parse_windows(DEFAULT_WINDOWS) if windows is None else windows
    raw_start = df.index[0]
    df = drop_listing_days(df, listing_days)
    r5, gaps5 = log_returns(df["close"], "5min")
    r1, _ = log_returns(df["close"], "1min")
    out = dict(symbol=symbol, start=raw_start, first=df.index[0], end=df.index[-1],
               gaps5=gaps5, n5=len(r5), nbars=len(df))
    out["vr_pooled"], out["vr_year"] = _vr_blocks(r5, reps, seed)
    out["vr_1m"] = vr_rows(r1.values, CONTAMINATED, BARS_PER_DAY_1M, reps, seed)
    out["hurst_year"] = {}
    for year, ry in r5.groupby(r5.index.year):
        if int(year) in out["vr_year"]:
            out["hurst_year"][int(year)] = dfa_hurst(ry.values, dfa_scales(12, 288))
    out["hurst_pooled"] = dfa_hurst(r5.values, dfa_scales(12, 288))
    out["range"] = {"1h": range_stats(df, "1h"), "4h": range_stats(df, "4h")}

    # Robust variant: drop event windows, then winsorise per calendar month.
    r5x = exclude_windows(r5, windows)
    r5w, n_capped = winsorise_monthly(r5x)
    dfx = exclude_windows(df, windows)
    rob = dict(n5=len(r5w), dropped=len(r5) - len(r5x), winsorised=n_capped)
    rob["vr_pooled"], rob["vr_year"] = _vr_blocks(r5w, reps, seed)
    rob["range"] = {"1h": range_stats(dfx, "1h"), "4h": range_stats(dfx, "4h")}
    out["robust"] = rob
    return out


RECENT_FROM = 2024


def holm_family(results: list[dict]) -> dict[tuple[str, str], float]:
    """Holm-adjusted robust pooled z* p-values; family = symbols x YEARLY_HORIZONS in this run."""
    keys, ps = [], []
    for res in results:
        pooled = {row["horizon"]: row for row in res["robust"]["vr_pooled"]}
        for h in YEARLY_HORIZONS:
            keys.append((res["symbol"], h))
            ps.append(pooled[h]["p"])
    return dict(zip(keys, holm_adjust(ps)))


def _year_ok(row: dict, robust: bool) -> bool:
    """Yearly check: CI excludes 1 (below); robust also needs unadjusted z* p < ALPHA."""
    return bool(row["sig_below"] and (not robust or row["p"] < ALPHA))


def evaluate_g0(results: list[dict], robust: bool = False) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Apply the G0 rule literally. Returns (passed, [(check, ok, detail)]).

    Baseline: CI excludes 1 (pooled) and in a majority of years.
    Robust (on the winsorised, window-excluded series): pooled also needs Holm-adjusted
    z* p < 0.05; yearly also needs the unadjusted z* p < 0.05 for that year.
    """
    checks = []
    holm = holm_family(results) if robust else {}
    for res in results:
        sym = res["symbol"]
        src = res["robust"] if robust else res
        pooled = {row["horizon"]: row for row in src["vr_pooled"]}
        for h in YEARLY_HORIZONS:
            row = pooled[h]
            detail = f"VR={row['vr']:.3f}, CI [{row['lo']:.3f}, {row['hi']:.3f}]"
            if robust:
                adj = holm[(sym, h)]
                ok = bool(row["sig_below"] and adj < ALPHA)
                detail += f", Holm p={adj:.4f}"
                name = f"{sym} pooled {h}: VR<1, CI excludes 1 and Holm p<{ALPHA}"
            else:
                ok = row["sig_below"]
                name = f"{sym} pooled {h}: VR<1 and CI excludes 1"
            checks.append((name, ok, detail))
            years = src["vr_year"]
            good = [y for y, rows in years.items()
                    for rw in rows if rw["horizon"] == h and _year_ok(rw, robust)]
            what = "CI excluding 1 and p<0.05" if robust else "CI excluding 1"
            checks.append((f"{sym} {h}: VR<1 with {what} in a majority of years",
                           len(good) * 2 > len(years), f"{len(good)} of {len(years)} years"))
        med = src["range"]["4h"]["median"]
        checks.append((f"{sym} median 4h range > {RANGE_THRESHOLD:.3%}", med > RANGE_THRESHOLD,
                       f"median {med:.3%}"))
    return all(ok for _, ok, _ in checks), checks


def symbol_verdicts(results: list[dict], robust: bool = False) -> list[dict]:
    """Per-symbol G0 verdict: every check for that symbol must hold.

    Also reports how many of the most recent years (>= RECENT_FROM) show
    significant mean reversion at 1h, as information (not part of the rule).
    """
    _, checks = evaluate_g0(results, robust)
    out = []
    for res in results:
        sym = res["symbol"]
        src = res["robust"] if robust else res
        mine = [(c, ok, d) for c, ok, d in checks if c.startswith(sym + " ")]
        recent = {y: rows for y, rows in src["vr_year"].items() if int(y) >= RECENT_FROM}
        recent_ok = sum(1 for rows in recent.values()
                        for rw in rows if rw["horizon"] == "1h" and _year_ok(rw, robust))
        out.append(dict(symbol=sym, passed=all(ok for _, ok, _ in mine),
                        failed=[c[len(sym) + 1:] for c, ok, _ in mine if not ok],
                        details={c[len(sym) + 1:]: d for c, _, d in mine},
                        recent=f"{recent_ok} of {len(recent)}"))
    return out


def final_verdicts(results: list[dict]) -> list[dict]:
    """Final = passes both the baseline and the robust version."""
    base = {v["symbol"]: v for v in symbol_verdicts(results)}
    rob = {v["symbol"]: v for v in symbol_verdicts(results, robust=True)}
    return [dict(symbol=r["symbol"], baseline=base[r["symbol"]]["passed"],
                 robust=rob[r["symbol"]]["passed"],
                 final=base[r["symbol"]]["passed"] and rob[r["symbol"]]["passed"],
                 robust_failed=rob[r["symbol"]]["failed"],
                 baseline_failed=base[r["symbol"]]["failed"]) for r in results]


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


def _verdict_table(results: list[dict], robust: bool) -> list[str]:
    lines = [f"| Coin | G0 | Pooled 1h | Pooled 4h | Years 1h | Years 4h | 1h since {RECENT_FROM} | "
             "Median 4h range |", "|---|---|---|---|---|---|---|---|"]
    for v in symbol_verdicts(results, robust):
        d = v["details"]
        get = lambda key: next((val for k, val in d.items() if k.startswith(key)), "")
        lines.append(f"| {v['symbol']} | {'**PASS**' if v['passed'] else 'fail'} | "
                     f"{get('pooled 1h')} | {get('pooled 4h')} | {get('1h:')} | {get('4h:')} | "
                     f"{v['recent']} | {get('median 4h')} |")
    return lines


def render_report(results: list[dict], reps: int, seed: int,
                  listing_days: float = LISTING_DAYS, windows=None) -> str:
    windows = parse_windows(DEFAULT_WINDOWS) if windows is None else windows
    passed, checks = evaluate_g0(results)
    rpassed, rchecks = evaluate_g0(results, robust=True)
    L = ["# G0 results: which coins mean-revert at grid horizons?", ""]
    for res in results:
        rob = res["robust"]
        L.append(f"- {res['symbol']}: data {res['start']:%Y-%m-%d} to {res['end']:%Y-%m-%d}, "
                 f"analysed from {res['first']:%Y-%m-%d}, {res['nbars']:,} 1m bars "
                 f"({res['gaps5']} empty 5m bins forward-filled); robust run dropped "
                 f"{rob['dropped']:,} 5m bars in excluded windows and winsorised "
                 f"{rob['winsorised']:,} of {rob['n5']:,}.")
    wtxt = "; ".join(f"{a:%Y-%m-%d %H:%M} to {b:%Y-%m-%d %H:%M} UTC" for a, b in windows)
    L += ["", f"Method: Lo-MacKinlay VR on log returns, robust z*, stationary block bootstrap "
          f"(mean block 1 day, {reps} reps, seed {seed}), 95% CI. The bootstrap percentile "
          "interval is re-centred on the point estimate: CI = VR + (q_lo - median(draws), "
          "q_hi - median(draws)), because plain percentile intervals drift toward 1 on short, "
          "heavy-tailed samples. "
          f"Each symbol's first {listing_days:g} days of data are excluded in both versions. "
          f"**Baseline**: the above on 5m returns as-is; pooled and yearly CI must exclude 1. "
          f"**Robust**: 5m returns winsorised at +-{WINSOR_K:g} sigma (sigma = {MAD_SCALE} x MAD, "
          f"per calendar month) and {wtxt} dropped; pooled checks need the Holm-adjusted z* p < "
          f"{ALPHA} (family = {len(results)} symbols x {{1h, 4h}} = {2 * len(results)} tests) "
          f"and a CI excluding 1; yearly checks need a CI excluding 1 and unadjusted z* p < "
          f"{ALPHA} for that year. Both use a majority-of-years rule and the median-4h-range "
          f"check. Year rows need at least {MIN_YEAR_DAYS} days of data. "
          "**Final** = passes both.", ""]

    L += ["## Final G1 universe (baseline and robust)", "",
          "| Coin | Baseline | Robust | Final (both) | Robust failures |", "|---|---|---|---|---|"]
    fin = final_verdicts(results)
    for v in fin:
        pf = lambda ok: "**PASS**" if ok else "fail"
        L.append(f"| {v['symbol']} | {pf(v['baseline'])} | {pf(v['robust'])} | {pf(v['final'])} | "
                 f"{'; '.join(v['robust_failed']) or '-'} |")
    keep = [v["symbol"] for v in fin if v["final"]]
    L += ["", f"Final list: {', '.join(keep) if keep else 'none'}. "
          f"Baseline passes {sum(v['baseline'] for v in fin)} of {len(fin)}, robust passes "
          f"{sum(v['robust'] for v in fin)} of {len(fin)}.", ""]

    L += ["## Verdict per coin: baseline", "",
          "Each coin is gated on its own. Only coins that pass both versions go on to the G1 replay.", ""]
    L += _verdict_table(results, False)
    L += ["", f"With {len(results)} coins tested, expect occasional false passes from chance; "
          "the majority-of-years rule limits this, and G1 (replay with fees) is the real test.", ""]

    L += ["## Verdict per coin: robust", "",
          "Winsorised, event windows excluded. Pooled cells show the Holm-adjusted z* p.", ""]
    L += _verdict_table(results, True)
    L.append("")
    L += ["Robust pooled and year counts:", "",
          "| Coin | Pooled 1h | Pooled 4h | Years 1h | Years 4h |", "|---|---|---|---|---|"]
    for v in symbol_verdicts(results, robust=True):
        d = v["details"]
        ok = lambda key: "pass" if not any(k.startswith(key) for k in v["failed"]) else "fail"
        get = lambda key: next((val.split(", Holm")[0] if "Holm" in val else val
                                for k, val in d.items() if k.startswith(key)), "")
        L.append(f"| {v['symbol']} | {ok('pooled 1h')}: {get('pooled 1h')} | "
                 f"{ok('pooled 4h')}: {get('pooled 4h')} | {get('1h:')} | {get('4h:')} |")
    L.append("")

    L += ["## Overall: baseline (all coins together)", ""]
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

    L += ["## Overall: robust (all coins together)", ""]
    if rpassed:
        L.append("**PASS.** Every robust check below holds.")
    else:
        failed = [c for c, ok, _ in rchecks if not ok]
        L.append(f"**FAIL.** {len(failed)} of {len(rchecks)} checks fail, e.g. {failed[0]}.")
    L += ["", "| Check | Result | Detail |", "|---|---|---|"]
    for c, ok, d in rchecks:
        L.append(f"| {c} | {'pass' if ok else 'fail'} | {d} |")
    L.append("")

    for res in results:
        L += [f"## {res['symbol']}: variance ratio, pooled (5m base)", ""]
        L += ["Baseline:", ""] + _vr_table(res["vr_pooled"])
        L += ["", "Robust (winsorised, event windows excluded):", ""]
        L += _vr_table(res["robust"]["vr_pooled"])
        L += ["", "1m base, labelled contaminated by bid-ask bounce (kline closes are trade "
              "prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); "
              "baseline only:", ""]
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
    L += ["", "## Stability: per calendar year, robust (VR, 95% CI, unadjusted z* p)", "",
          "| Symbol | Year | 1h VR | 1h CI | 1h p | 4h VR | 4h CI | 4h p |", "|---|---|---|---|---|---|---|---|"]
    for res in results:
        for year, rows in res["robust"]["vr_year"].items():
            d = {r["horizon"]: r for r in rows}
            a, b = d["1h"], d["4h"]
            L.append(f"| {res['symbol']} | {year} | {a['vr']:.3f} | [{a['lo']:.3f}, {a['hi']:.3f}] | "
                     f"{a['p']:.3f} | {b['vr']:.3f} | [{b['lo']:.3f}, {b['hi']:.3f}] | {b['p']:.3f} |")
    L += ["", "## DFA Hurst exponent (5m returns, 1h-1d scales; baseline)", "",
          "| Symbol | Pooled H |", "|---|---|"]
    for res in results:
        L.append(f"| {res['symbol']} | {res['hurst_pooled']:.3f} |")
    L += ["", "H < 0.5 means mean reversion, 0.5 a random walk, > 0.5 trending.", ""]

    L += ["## Range vs fees: (high - low) / open", "",
          f"Maker round trip {MAKER_ROUND_TRIP:.2%} (BNB); 2.5x threshold {RANGE_THRESHOLD:.3%}. "
          "Robust rows exclude the event windows.", "",
          "| Symbol | Bar | Median | 25th pct | Median vs 0.375% | Robust median |", "|---|---|---|---|---|---|"]
    for res in results:
        for bar in ("1h", "4h"):
            s = res["range"][bar]
            L.append(f"| {res['symbol']} | {bar} | {s['median']:.3%} | {s['p25']:.3%} | "
                     f"{'above' if s['median'] > RANGE_THRESHOLD else 'below'} | "
                     f"{res['robust']['range'][bar]['median']:.3%} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT"])
    ap.add_argument("--out", default="docs/research/G0_RESULTS.md")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--exclude-listing-days", type=float, default=LISTING_DAYS,
                    help="drop each symbol's first N days of data (default %(default)s)")
    ap.add_argument("--exclude-window", nargs=2, action="append", metavar=("START", "END"),
                    help="robust variant: drop 5m bars in [START, END) UTC; repeatable "
                         f"(default {DEFAULT_WINDOWS[0][0]} {DEFAULT_WINDOWS[0][1]})")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    windows = parse_windows(args.exclude_window or DEFAULT_WINDOWS)
    results = []
    for sym in args.symbols:
        print(f"analysing {sym} ...", flush=True)
        results.append(analyse_symbol(sym, load_klines(sym, Path(args.data_dir)),
                                      args.reps, args.seed, args.exclude_listing_days, windows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(results, args.reps, args.seed,
                                      args.exclude_listing_days, windows))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
