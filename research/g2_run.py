"""G2 runner: trend + vol targeting, spot (S1, S2) and USD-M perpetuals (P1, P2).

Everything comes from ``research/g2_preregistration.json`` (read-only). Strategy maths are
``vv.core.trend``, execution is ``vv.sim.portfolio_sim``, metrics are ``vv.sim.metrics``, the
dial features are built by ``research.feature_screen`` (its lagging rules).

Usage: .venv/bin/python -m research.g2_run [--processes 3] [--out docs/research/G2_RESULTS.md]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from research import feature_screen as fs
from vv.core.trend import TrendParams, apply_band, risk_dial, target_weights
from vv.sim import metrics as mx
from vv.sim.portfolio_sim import simulate

ROOT = Path(__file__).resolve().parent.parent
PREREG = json.loads((ROOT / "research" / "g2_preregistration.json").read_text())
SYMS: list[str] = PREREG["universe"]
START = pd.Timestamp(PREREG["period"]["start"])
END = pd.Timestamp(PREREG["period"]["end"])
VARIANTS: dict[str, dict] = PREREG["variants"]
GATE = PREREG["gate_g2_per_variant"]
SPLITS = PREREG["splits"]
SPLIT_RANGE = {"2020": ("2020-01-01", "2020-12-31"), "2021": ("2021-01-01", "2021-12-31"),
               "2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"),
               "2024": ("2024-01-01", "2024-12-31"), "2025-2026": ("2025-01-01", "2026-10-31")}
OOS_NAMES = [s["name"] for s in SPLITS if s["role"] == "out_of_sample"]
START_CASH = 10_000.0
DATA = ROOT / "data" / "futures"
STATE: dict = {}


# ----------------------------------------------------------------------------- data
def daily_close_from_spot(spot_1h: pd.Series) -> pd.Series:
    """Spot 1h closes are labelled by close time; the close of UTC day d is the value at d+1 00:00."""
    day = (spot_1h.index - pd.Timedelta(nanoseconds=1)).floor("D")
    return spot_1h.groupby(day).last()


def load_data() -> dict:
    spot, perp, funding = {}, {}, {}
    daily = {}
    for s in SYMS:
        sp = pd.read_parquet(DATA / f"{s}_spot_1h.parquet")["close"].dropna()
        spot[s] = pd.DataFrame({"close": sp})
        perp[s] = pd.read_parquet(DATA / f"{s}_perp_1h.parquet")[["open", "high", "low", "close"]]
        funding[s] = pd.read_parquet(DATA / f"{s}_funding.parquet")["rate"]
        daily[s] = daily_close_from_spot(sp)
    cal = pd.date_range("2019-01-01", END, freq="D")
    daily_close = pd.DataFrame(daily).reindex(cal)
    inp = fs.load_inputs(ROOT / "data")
    feats = fs.build_features(inp, pd.date_range("2019-01-01", END, freq="D", tz="UTC"))["basket"]
    feats = feats[["stablecoin_supply_chg_30d", "vix_level", "global_ls_ratio"]]
    feats.index = feats.index.tz_localize(None)
    return {"spot": spot, "perp": perp, "funding": funding, "daily_close": daily_close, "features": feats}


def build_weights(variant: str) -> pd.DataFrame:
    cfg = VARIANTS[variant]
    dc = STATE["daily_close"]
    params = TrendParams(allow_short=cfg["allow_short"])
    dial = risk_dial(STATE["features"], params.dial_floor) if cfg["risk_dial"] else None
    w = target_weights(dc, params, dial)
    w = w.loc[START - pd.Timedelta(days=1): END - pd.Timedelta(days=1)]
    if cfg["venue"] == "perp":  # a perp must be listed at the execution bar
        for s in SYMS:
            first = STATE["perp"][s].index[0]
            w.loc[(w.index + pd.Timedelta(days=1, hours=1)) < first, s] = 0.0
    return w


def run_variant(job: tuple[str, float]):
    variant, mult = job
    cfg = VARIANTS[variant]
    venue = cfg["venue"]
    costs = PREREG["costs"][venue]
    params = TrendParams(allow_short=cfg["allow_short"])
    prices = STATE[venue]
    res = simulate(
        build_weights(variant), venue, {s: prices[s] for s in SYMS},
        STATE["funding"] if venue == "perp" else {},
        fee=costs["taker"] * mult, slippage_bps=PREREG["costs"]["slippage_bps"] * mult,
        maintenance_margin=costs.get("maintenance_margin", 0.005), start_cash=START_CASH,
        band=lambda cur, tgt: apply_band(cur, tgt, params),
    )
    return variant, mult, res


# ----------------------------------------------------------------------------- metrics
def window(eq: pd.Series, lo: str, hi: str) -> pd.Series:
    """Equity over [lo, hi] including the prior day's close as the base point."""
    lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
    return eq.loc[lo_t - pd.Timedelta(days=1): hi_t]


def split_table(res) -> pd.DataFrame:
    rows = []
    eq = res.equity_daily
    for name, (lo, hi) in SPLIT_RANGE.items():
        w = window(eq, lo, hi)
        r = mx.daily_returns(w)
        rows.append({"split": name, "role": next(s["role"] for s in SPLITS if s["name"] == name),
                     "return": mx.total_return(w), "sharpe": mx.sharpe(r), "max_dd": mx.max_drawdown(w),
                     "excess_cash": mx.total_return(w) - mx.cash_yield_return(w.index)})
    return pd.DataFrame(rows)


def oos_equity(eq: pd.Series) -> pd.Series:
    return window(eq, SPLIT_RANGE[OOS_NAMES[0]][0], SPLIT_RANGE[OOS_NAMES[-1]][1])


def exposure_shares(res) -> dict[str, float]:
    e = res.exposure_daily.loc[START:END]
    longs, shorts = e.clip(lower=0).sum(axis=1), (-e.clip(upper=0)).sum(axis=1)
    lg, sh = longs > 0.01, shorts > 0.01
    return {"long": float((lg & ~sh).mean()), "short": float((~lg & sh).mean()),
            "both": float((lg & sh).mean()), "flat": float((~lg & ~sh).mean())}


def summarise(res, res_stress) -> dict:
    eq = res.equity_daily
    full = eq.loc[START - pd.Timedelta(days=1):]
    oos = oos_equity(eq)
    oos_s = oos_equity(res_stress.equity_daily)
    r_oos, r_oos_s = mx.daily_returns(oos), mx.daily_returns(oos_s)
    lo, hi = mx.sharpe_ci(r_oos_s, mean_block=10, reps=2000, seed=0)
    st = split_table(res)
    return {
        "splits": st,
        "oos_return": mx.total_return(oos), "oos_cagr": mx.cagr(oos), "oos_sharpe": mx.sharpe(r_oos),
        "oos_maxdd": mx.max_drawdown(oos), "oos_excess_cash": mx.total_return(oos) - mx.cash_yield_return(oos.index),
        "oos_cagr_stress": mx.cagr(oos_s), "oos_sharpe_stress": mx.sharpe(r_oos_s), "ci_lo": lo, "ci_hi": hi,
        "full_return": mx.total_return(full), "full_cagr": mx.cagr(full), "full_sharpe": mx.sharpe(mx.daily_returns(full)),
        "full_maxdd": mx.max_drawdown(full), "maxdd_stress": mx.max_drawdown(res_stress.equity_daily.loc[START - pd.Timedelta(days=1):]),
        "fees": res.fees_paid, "funding": res.funding_paid, "turnover": res.turnover, "max_gross": res.max_gross,
        "liqs": len(res.liquidations), "liqs_stress": len(res_stress.liquidations),
        "shares": exposure_shares(res), "warnings": res.coverage_warnings,
        "final_equity": float(full.iloc[-1]), "final_equity_stress": float(res_stress.equity_daily.iloc[-1]),
    }


def gate(m: dict) -> list[tuple[str, str, bool]]:
    st = m["splits"]
    pos = int((st["return"] > 0).sum())
    pos_2022 = bool(st.loc[st["split"] == "2022", "return"].iloc[0] > 0)
    return [
        ("OOS Sharpe 95% CI lower bound > 0 (stress: fees x2, slippage x2)",
         f"{m['ci_lo']:.2f} (CI {m['ci_lo']:.2f} to {m['ci_hi']:.2f})", m["ci_lo"] > GATE["oos_sharpe_ci_lower_gt"]),
        ("OOS annualised return > 4% (pre-registered costs)", f"{m['oos_cagr']:.1%}", m["oos_cagr"] > 0.04),
        ("Net return > 0 in a majority of 6 splits, 2022 among them", f"{pos}/6, 2022 {'+' if pos_2022 else '-'}",
         pos > len(st) / 2 and pos_2022),
        (f"Full-run max drawdown <= {GATE['max_drawdown_full_run']:.0%}", f"{m['full_maxdd']:.1%}",
         m["full_maxdd"] <= GATE["max_drawdown_full_run"]),
        ("No liquidations", f"{m['liqs']} (stress {m['liqs_stress']})", m["liqs"] == 0),
    ]


def benchmarks(dc: pd.DataFrame) -> dict[str, pd.Series]:
    """Equal-weight buy-and-hold of the coins with >=120 days of spot history at the start (no
    rebalancing; later listings ignored), and cash at 4% a year, on the sim's daily labels."""
    idx = pd.date_range(START - pd.Timedelta(days=1), END, freq="D")
    c = dc.reindex(idx)
    elig = [s for s in SYMS if dc[s].first_valid_index() is not None
            and dc[s].first_valid_index() <= START - pd.Timedelta(days=120)]
    units = {s: START_CASH / len(elig) / c[s].iloc[0] for s in elig}
    bh = sum(units[s] * c[s].ffill() for s in elig)
    return {"equal_weight_bh": bh.rename("bh"), "cash_4pct": mx.cash_yield_curve(idx, START_CASH)}


# ----------------------------------------------------------------------------- report
def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render(results: dict, bench: dict, gates: dict, elapsed: float) -> str:
    names = list(VARIANTS)
    passed = [v for v in names if all(g[2] for g in gates[v])]
    oos_rows = []
    for v in names:
        m = results[v]
        oos_rows.append([f"{v} ({VARIANTS[v]['venue']}{', dial' if VARIANTS[v]['risk_dial'] else ''})",
                         pct(m["oos_cagr"]), f"{m['oos_sharpe']:.2f}", pct(m["oos_excess_cash"]), pct(m["oos_maxdd"]),
                         pct(m["oos_cagr_stress"]), f"{m['oos_sharpe_stress']:.2f}",
                         f"{m['ci_lo']:.2f} / {m['ci_hi']:.2f}"])
    for k, label in (("equal_weight_bh", "Equal-weight buy-and-hold"), ("cash_4pct", "Cash at 4%")):
        o = oos_equity(bench[k])
        sh = "-" if k == "cash_4pct" else f"{mx.sharpe(mx.daily_returns(o)):.2f}"
        oos_rows.append([label, pct(mx.cagr(o)), sh,
                         pct(mx.total_return(o) - mx.cash_yield_return(o.index)), pct(mx.max_drawdown(o)), "-", "-", "-"])
    lines = ["# G2 results: trend + vol targeting, spot vs perpetuals", "",
             f"Pre-registration `research/g2_preregistration.json` (unchanged). One continuous run per variant, "
             f"2020-01-01 to 2026-10-31, 10,000 USDT start; splits are reporting windows. OOS = 2024 to 2026-10-31 "
             f"(seen by the team in other contexts; disclosed). Generated by `research/g2_run.py` in {elapsed / 60:.1f} min.", "",
             "## Out-of-sample, side by side", "",
             md_table(["Variant", "OOS ann. return", "OOS Sharpe", "Excess over 4% cash (total)", "OOS max DD",
                       "Stress ann. return", "Stress Sharpe", "Stress Sharpe CI95"], oos_rows), "",
             "## Gate G2 per variant", ""]
    for v in names:
        lines += [f"**{v}: {'PASS' if all(g[2] for g in gates[v]) else 'FAIL'}**", "",
                  md_table(["Criterion", "Value", "Result"], [[a, b, "pass" if c else "FAIL"] for a, b, c in gates[v]]), ""]
    lines += ["## Full-run costs and exposure", "",
              md_table(["Variant", "Final equity", "Full CAGR", "Full Sharpe", "Full max DD", "Fees", "Funding (+paid)",
                        "Turnover /yr", "Max gross", "Long / short / both / flat (days)", "Coverage warnings"],
                       [[v, f"{results[v]['final_equity']:.0f}", pct(results[v]["full_cagr"]),
                         f"{results[v]['full_sharpe']:.2f}", pct(results[v]["full_maxdd"]), f"{results[v]['fees']:.0f}",
                         f"{results[v]['funding']:.0f}", f"{results[v]['turnover']:.1f}x", f"{results[v]['max_gross']:.2f}",
                         " / ".join(pct(results[v]["shares"][k], 0) for k in ("long", "short", "both", "flat")),
                         str(results[v]["warnings"])] for v in names]), ""]
    bh_full = bench["equal_weight_bh"]
    lines += [f"Benchmark equal-weight buy-and-hold: full-run return {pct(mx.total_return(bh_full))}, "
              f"max DD {pct(mx.max_drawdown(bh_full))}. Cash at 4%: {pct(mx.total_return(bench['cash_4pct']))}.", "",
              "## Per-split results (pre-registered costs)", ""]
    for v in names:
        st = results[v]["splits"]
        lines += [f"**{v}**", "", md_table(["Split", "Role", "Return", "Sharpe", "Max DD", "Excess over cash"],
                  [[r.split, r.role, pct(r["return"]), f"{r.sharpe:.2f}", pct(r.max_dd), pct(r.excess_cash)]
                   for _, r in st.iterrows()]), ""]
    lines += ["## Method notes", "",
              "Gate criteria use the pre-registered-cost run, except the Sharpe CI (stress run, as pre-registered). "
              "Perp variants take signals from spot closes, are zeroed before the perp lists, and a coin with no bar "
              "at an execution instant is held (counted as a coverage warning). Execution is the first hourly instant "
              "after the daily close (perp open, spot close) with 5 bp slippage and taker fees.", "",
              "## Verdict", "", verdict(results, gates, passed), "", "## What futures added (P vs S, same signal)", "",
              futures_note(results), ""]
    return "\n".join(lines)


def verdict(results, gates, passed) -> str:
    if passed:
        return (f"Variants passing every G2 criterion: {', '.join(passed)}. Per the pre-registration a pass only "
                "earns paper trading as the true holdout, not capital.")
    fails = {v: [a.split(" (")[0] for a, _, ok in gates[v] if not ok] for v in gates}
    parts = "; ".join(f"{v} fails: {', '.join(f)}" for v, f in fails.items())
    return ("No variant passes G2 as pre-registered, so nothing proceeds to paper trading from this run. "
            f"{parts}. The gate was applied literally and is not to be re-tuned after seeing these numbers.")


def futures_note(results) -> str:
    out = []
    for s, p, label in (("S1", "P1", "no dial"), ("S2", "P2", "with dial")):
        a, b = results[s], results[p]
        out.append(f"- {label}: OOS ann. return {pct(a['oos_cagr'])} (S) vs {pct(b['oos_cagr'])} (P); OOS Sharpe "
                   f"{a['oos_sharpe']:.2f} vs {b['oos_sharpe']:.2f}; full-run max DD {pct(a['full_maxdd'])} vs "
                   f"{pct(b['full_maxdd'])}; funding on P {b['funding']:+.0f} USDT (positive = paid); P was short on "
                   f"{pct(b['shares']['short'] + b['shares']['both'], 0)} of days; stress CI lower bound "
                   f"{a['ci_lo']:.2f} (S) vs {b['ci_lo']:.2f} (P).")
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--out", default="docs/research/G2_RESULTS.md")
    args = ap.parse_args(argv)
    procs = max(1, min(args.processes, 3))
    t0 = time.time()
    STATE.update(load_data())
    jobs = [(v, m) for v in VARIANTS for m in (1.0, 2.0)]
    print(f"running {len(jobs)} sims on {procs} processes", file=sys.stderr)
    with get_context("fork").Pool(procs) as pool:
        out = pool.map(run_variant, jobs)
    sims = {(v, m): r for v, m, r in out}
    results = {v: summarise(sims[(v, 1.0)], sims[(v, 2.0)]) for v in VARIANTS}
    gates = {v: gate(results[v]) for v in VARIANTS}
    bench = benchmarks(STATE["daily_close"])
    for v in VARIANTS:
        print(v, "PASS" if all(g[2] for g in gates[v]) else "FAIL",
              f"oos_cagr={results[v]['oos_cagr']:.3f} sharpe={results[v]['oos_sharpe']:.2f} ci_lo={results[v]['ci_lo']:.2f}",
              file=sys.stderr)
    text = render(results, bench, gates, time.time() - t0)
    (ROOT / args.out).write_text(text)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
