"""G1 runner: replay the production strategy on historical data under the pre-registered plan.

Usage:
    python -m research.g1_run --symbols BTCUSDT SOLUSDT XRPUSDT DOGEUSDT LINKUSDT \
        --prereg research/g1_preregistration.json --out docs/research/G1_RESULTS.md
    python -m research.g1_run --symbols BTCUSDT SOLUSDT --smoke --out /tmp/g1_smoke.md

Flow (all settings come from the pre-registration file, which is never modified here):
  1. every trial x split on a fresh shared-wallet portfolio (cash reset, halt cleared);
  2. selection on in-sample splits only: highest median net Sharpe;
  3. the selected trial under the stress variants (fees x2, fill_haircut 0.7) on all splits;
  4. the shadow control (ETH) as its own portfolio with the selected trial;
  5. ``gate_g1`` applied literally; 6. a short markdown report.

Interpretations the pre-registration leaves open (documented in the report):
  * net capture over fees = pooled gross realised PnL / pooled fees over all splits, base costs;
  * positive splits and max drawdown: base costs, every split run (drawdown = worst split);
  * OOS Sharpe CI: daily returns of the out-of-sample splits concatenated, fees x2 runs;
  * the strategy's own ``maker_fee`` stays at its pre-registered value in stress runs: only
    the exchange side is stressed (fees x2 keeps the 10 bps taker slippage).

``--smoke`` is a plumbing check (2 symbols, 1 trial, Jan-Feb 2024, trimmed history); it is not
a G1 result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vv.core.params import GridParams, RiskParams
from vv.core.types import SymbolFilters
from vv.sim import metrics as mx
from vv.sim.replay import Fees, ReplayJob, ReplayResult, precompute_features, run_many

BOOT_SEED = 20261103
REPLAY_SEED = 0
SMOKE_WARMUP = pd.Timedelta(days=92)
RESULTS_PATH = "docs/research/G1_RESULTS.md"

VARIANTS: dict[str, tuple[float, float]] = {  # name -> (fee multiplier, fill_haircut)
    "base": (1.0, 1.0),
    "fees2": (2.0, 1.0),
    "haircut": (1.0, 0.7),
    "haircut_fees2": (2.0, 0.7),
}

# Binance-style filters (tick, step, min_qty, min_notional). Assumption: fixed over the whole
# history; DOGE/LINK use finer ticks than today's so early sub-$1 prices are not distorted.
_FILTERS = {
    "BTCUSDT": (0.01, 1e-5, 1e-5, 5.0),
    "ETHUSDT": (0.01, 1e-4, 1e-4, 5.0),
    "BNBUSDT": (0.01, 1e-3, 1e-3, 5.0),
    "SOLUSDT": (0.001, 1e-3, 1e-3, 5.0),
    "XRPUSDT": (1e-4, 0.1, 0.1, 5.0),
    "DOGEUSDT": (1e-6, 1.0, 1.0, 5.0),
    "LINKUSDT": (1e-4, 1e-2, 1e-2, 5.0),
    "AVAXUSDT": (0.01, 1e-2, 1e-2, 5.0),
}


def filters_for(symbol: str, bars: pd.DataFrame) -> SymbolFilters:
    if symbol in _FILTERS:
        return SymbolFilters(*_FILTERS[symbol])
    med = float(bars["close"].median())  # unknown symbol: tick/step ~1e-5 of the median price
    tick = 10.0 ** (math.floor(math.log10(med)) - 5)
    return SymbolFilters(tick, tick * 10, tick * 10, 5.0)


def load_klines(symbol: str, data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(data_dir) / "klines" / f"{symbol}_1m.parquet",
                         columns=["open", "high", "low", "close"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


# --------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class Trial:
    idx: int
    min_step: float
    spacing_sigmas: float

    @property
    def label(self) -> str:
        return f"T{self.idx + 1:02d} step {self.min_step:.4f} / {self.spacing_sigmas:g}s"


@dataclass(frozen=True)
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    role: str  # in_sample | out_of_sample

    @property
    def days(self) -> float:
        return (self.end - self.start) / pd.Timedelta(days=1)


def _utc(x: str) -> pd.Timestamp:
    return pd.Timestamp(x, tz="UTC")


def build_trials(prereg: dict) -> list[Trial]:
    t = prereg["trials"]
    pairs = [(m, s) for m in t["min_step"] for s in t["spacing_sigmas"]]
    return [Trial(i, ms, sp) for i, (ms, sp) in enumerate(pairs)]


def build_splits(prereg: dict, smoke: bool) -> list[Split]:
    if smoke:
        return [
            Split("smoke-Jan", _utc("2024-01-01"), _utc("2024-02-01"), "in_sample"),
            Split("smoke-Feb", _utc("2024-02-01"), _utc("2024-03-01"), "out_of_sample"),
        ]
    return [Split(s["name"], _utc(s["start"]), _utc(s["end"]), s["role"]) for s in prereg["splits"]]


def grid_for(prereg: dict, trial: Trial) -> GridParams:
    g = prereg["fixed_grid"]
    return GridParams(
        spacing_sigmas=trial.spacing_sigmas, min_step=trial.min_step,
        sigma_horizon_minutes=g["sigma_horizon_minutes"], levels=g["levels"],
        qty_multiplier=g["qty_multiplier"], tp_fee_multiple=g["tp_fee_multiple"],
        maker_fee=g["maker_fee"],
    )


def fees_for(prereg: dict, mult: float) -> Fees:
    f = prereg["fees"]
    return Fees(f["maker"] * mult, f["taker"] * mult, f["taker_slippage_bps"])


# --------------------------------------------------------------------------- stats / gate
@dataclass
class Stat:
    split: str
    role: str
    summary: dict
    returns: np.ndarray
    rejections: dict[str, int]
    n_places: int
    n_cancels: int
    n_decisions: int
    halts: int
    time_in_market: float
    max_exposure_total_post: float
    elapsed: float
    symbols: tuple[str, ...]
    final_equity: float
    daily_equity: pd.Series = field(repr=False, default=None)  # type: ignore[assignment]


def stat_of(split: Split, res: ReplayResult) -> Stat:
    summ = mx.summarize(
        res.daily_equity, fees_paid=res.fees_paid, gross_pnl=res.gross_pnl,
        round_trips=res.round_trips, avg_exposure=res.avg_exposure,
    )
    return Stat(
        split.name, split.role, summ, mx.daily_returns(res.daily_equity).to_numpy(),
        dict(res.rejections), res.n_places, res.n_cancels, res.n_decisions,
        len(res.halt_events), res.time_in_market, res.max_exposure_total_post, res.elapsed,
        res.symbols, res.final_equity, res.daily_equity,
    )


def select_trial(trials: list[Trial], stats: dict[tuple[int, str], Stat], splits: list[Split]) -> int:
    """Highest median net Sharpe across in-sample splits (first trial on ties)."""
    ins = [s.name for s in splits if s.role == "in_sample"]
    scores = [float(np.median([stats[(t.idx, n)].summary["sharpe"] for n in ins])) for t in trials]
    return int(np.argmax(scores))  # argmax returns the first maximum


def gate_g1(
    gate: dict,
    splits: list[Split],
    base: dict[str, Stat],
    fees2: dict[str, Stat],
    boot_reps: int = 2000,
) -> list[dict]:
    """Apply the pre-registered gate literally; one row per criterion."""
    rows: list[dict] = []
    gross = sum(base[s.name].summary["gross_pnl"] for s in splits)
    fees = sum(base[s.name].summary["fees_paid"] for s in splits)
    cap = mx.net_capture_over_fees(gross, fees)
    rows.append(dict(
        name="Net capture over fees", rule=f">= {gate['net_capture_over_fees_min']}",
        value=f"{cap:.2f}", passed=cap >= gate["net_capture_over_fees_min"],
        note="pooled gross realised / fees, all splits, base costs",
    ))

    oos = [s for s in splits if s.role == "out_of_sample"]
    r = np.concatenate([fees2[s.name].returns for s in oos]) if oos else np.array([])
    lo, hi = mx.sharpe_ci(r, mean_block=10, reps=boot_reps, seed=BOOT_SEED)
    rows.append(dict(
        name="OOS Sharpe CI lower bound", rule="> 0",
        value=f"{lo:.2f} (CI {lo:.2f}..{hi:.2f}, point {mx.sharpe(r):.2f})",
        passed=bool(lo > 0), note="fees x2, OOS splits pooled, stationary bootstrap, block 10d",
    ))

    pos = [s.name for s in splits if base[s.name].summary["total_return"] > 0]
    ok = len(pos) * 2 > len(splits) and "2022" in pos
    rows.append(dict(
        name="Positive splits", rule="majority and 2022 among them",
        value=f"{len(pos)}/{len(splits)} ({', '.join(pos) or 'none'}); 2022 {'in' if '2022' in pos else 'not in'}",
        passed=bool(ok), note="net return > 0, base costs",
    ))

    worst = max(base[s.name].summary["max_drawdown"] for s in splits)
    rows.append(dict(
        name="Max drawdown", rule=f"<= {gate['max_drawdown_max']:.0%}",
        value=f"{worst:.1%}", passed=bool(worst <= gate["max_drawdown_max"]),
        note="worst split, base costs",
    ))
    return rows


def gate_passes(rows: list[dict]) -> bool:
    return all(r["passed"] for r in rows)


# --------------------------------------------------------------------------- orchestration
def make_jobs(
    prereg: dict, trial: Trial, splits: list[Split], variant: str, symbols: tuple[str, ...],
    tag: str,
) -> list[ReplayJob]:
    mult, haircut = VARIANTS[variant]
    risk = RiskParams(**prereg["risk"])
    grid = grid_for(prereg, trial)
    fees = fees_for(prereg, mult)
    jobs = []
    for sp in sorted(splits, key=lambda s: -s.days):  # long jobs first
        jobs.append(ReplayJob(
            key=(tag, trial.idx, sp.name, variant),
            kwargs=dict(
                grid=grid, risk=risk, start=sp.start, end=sp.end,
                decision_minutes=prereg["decision_interval_minutes"],
                start_cash=prereg["start_cash"], fees=fees, fill_haircut=haircut,
                seed=REPLAY_SEED, keep_fills=False,
            ),
            symbols=symbols,
        ))
    return jobs


def run_set(jobs, splits, shared, processes, label) -> dict[tuple, Stat]:
    t0 = time.time()
    by_name = {s.name: s for s in splits}
    res = run_many(jobs, processes=processes, **shared)
    out = {k: stat_of(by_name[k[2]], r) for k, r in res.items()}
    print(f"  {label}: {len(jobs)} runs in {time.time() - t0:.0f}s", file=sys.stderr)
    return out


def symbol_days(bars: dict[str, pd.DataFrame], splits: list[Split], symbols) -> float:
    """Symbol-days of data inside the splits (for runtime projection)."""
    total = 0.0
    for s in symbols:
        idx = bars[s].index
        for sp in splits:
            lo = max(sp.start, idx[0])
            hi = min(sp.end, idx[-1] + pd.Timedelta(minutes=1))
            total += max(0.0, (hi - lo) / pd.Timedelta(days=1))
    return total


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def render(
    *, prereg, smoke, trials, splits, symbols, stats, sel, stress, shadow, shadow_gate,
    gate_rows, rerun_rows, baselines, runtime, projection,
) -> str:
    L: list[str] = []
    sel_t = trials[sel]
    ins = [s.name for s in splits if s.role == "in_sample"]
    title = "G1 results" + (" (SMOKE RUN, plumbing check only, not a G1 result)" if smoke else "")
    L += [f"# {title}", ""]
    L += [
        f"Universe: {', '.join(symbols)} (one shared wallet, {prereg['start_cash']:,} USDT). "
        f"Pre-registration v{prereg['version']}; decisions every {prereg['decision_interval_minutes']} min; "
        f"trial count = {len(trials)}. Runtime {runtime:.0f}s.",
        "",
    ]

    L += ["## Trials (in-sample only; selection = highest median net Sharpe)", ""]
    rows = []
    for t in trials:
        ss = [stats[(t.idx, n)] for n in ins]
        rows.append([
            t.label + (" **selected**" if t.idx == sel else ""),
            f"{np.median([x.summary['sharpe'] for x in ss]):.2f}",
            pct(float(np.median([x.summary['total_return'] for x in ss]))),
            pct(max(x.summary["max_drawdown"] for x in ss)),
            f"{sum(x.summary['round_trips'] for x in ss)}",
            f"{mx.net_capture_over_fees(sum(x.summary['gross_pnl'] for x in ss), sum(x.summary['fees_paid'] for x in ss)):.2f}",
        ])
    L += md_table(["trial", "median Sharpe", "median return", "worst MDD", "round trips", "capture/fees"], rows)

    L += ["", f"## Selected trial {sel_t.label} vs baselines (base costs)", ""]
    rows = []
    for sp in splits:
        s = stats[(sel, sp.name)]
        b = baselines[sp.name]
        rows.append([
            f"{sp.name} ({'IS' if sp.role == 'in_sample' else 'OOS'})",
            pct(s.summary["total_return"]), f"{s.summary['sharpe']:.2f}", pct(s.summary["max_drawdown"]),
            f"{s.summary['net_capture_over_fees']:.2f}", pct(s.summary["avg_exposure"]),
            pct(b["equal_weight_bh"]["total_return"]), pct(b["equal_weight_bh"]["max_drawdown"]),
            pct(b["bh_scaled_to_exposure"]["total_return"]), "0.0%",
        ])
    L += md_table(
        ["split", "return", "Sharpe", "MDD", "capture", "avg exp", "B&H ret", "B&H MDD", "B&H@exp ret", "cash"],
        rows,
    )

    L += ["", "## Stress (selected trial)", ""]
    rows = []
    for variant, label in (("fees2", "fees x2"), ("haircut", "fill_haircut 0.7")):
        for sp in splits:
            s = stress[(sel, sp.name, variant)]
            rows.append([label, sp.name, pct(s.summary["total_return"]), f"{s.summary['sharpe']:.2f}",
                         pct(s.summary["max_drawdown"]), f"{s.summary['net_capture_over_fees']:.2f}"])
    L += md_table(["variant", "split", "return", "Sharpe", "MDD", "capture"], rows)

    L += ["", f"## Shadow control ({', '.join(prereg['shadow_control'])}, own portfolio, selected trial)", ""]
    rows = []
    for sp in splits:
        s = shadow[(sp.name, "base")]
        rows.append([sp.name, pct(s.summary["total_return"]), f"{s.summary['sharpe']:.2f}",
                     pct(s.summary["max_drawdown"]), f"{s.summary['net_capture_over_fees']:.2f}"])
    L += md_table(["split", "return", "Sharpe", "MDD", "capture"], rows)
    eth_pass = gate_passes(shadow_gate)
    L += ["", "ETH gate: " + ("PASS. **Flag: the fill model may be too generous.**" if eth_pass else "FAIL (the control behaves as expected)."), ""]
    L += md_table(["criterion", "ETH value", "result"], [[r["name"], r["value"], "PASS" if r["passed"] else "FAIL"] for r in shadow_gate])

    L += ["", "## Gate G1", ""]
    L += md_table(
        ["criterion", "rule", "value", "result"],
        [[r["name"], r["rule"], r["value"], "PASS" if r["passed"] else "FAIL"] for r in gate_rows],
    )
    if rerun_rows is not None:
        L += ["", "Rerun at fill_haircut 0.7 (triggered by the ETH shadow rule):", ""]
        L += md_table(
            ["criterion", "rule", "value", "result"],
            [[r["name"], r["rule"], r["value"], "PASS" if r["passed"] else "FAIL"] for r in rerun_rows],
        )
    failed = [r["name"] for r in gate_rows if not r["passed"]]
    if smoke:
        verdict = "Smoke run: not a verdict. The 2022 criterion cannot be met without the 2022 split."
    elif not failed:
        verdict = "PASS on the literal gate."
        if eth_pass:
            verdict += " The zero-capital ETH control also passed, so the pass is not trusted until the haircut-0.7 rerun above passes too."
    else:
        verdict = "FAIL. Criteria not met: " + ", ".join(failed) + ". The strategy does not clear G1 as pre-registered."
    L += ["", f"**Verdict.** {verdict}", ""]

    all_rej: Counter[str] = Counter()
    for s in stats.values():
        all_rej.update(s.rejections)
    sel_rej: Counter[str] = Counter()
    for sp in splits:
        sel_rej.update(stats[(sel, sp.name)].rejections)
    sel_all = [stats[(sel, sp.name)] for sp in splits]
    n_dec = sum(s.n_decisions for s in sel_all)
    L += ["## Notes", ""]
    L += [
        f"- Trials run: {len(trials)} x {len(splits)} splits = {len(trials) * len(splits)} base runs; "
        f"stress and shadow runs reuse the selected trial.",
        f"- Order rejections, selected trial base runs: {dict(sel_rej) or 'none'}; all {len(stats)} base runs: {dict(all_rej) or 'none'}.",
        f"- Churn (selected, base): {sum(s.n_places for s in sel_all)} placed, {sum(s.n_cancels for s in sel_all)} cancelled over {n_dec} decisions; "
        f"halts {sum(s.halts for s in sel_all)}; peak exposure incl. resting buys {pct(max(s.max_exposure_total_post for s in sel_all))} of equity.",
        "- Interpretations: capture = pooled gross realised / fees (all splits); drawdown = worst split; "
        "OOS CI = pooled OOS daily returns at fees x2; stress changes exchange fees only, not the strategy's maker_fee input; "
        "baselines: equal-weight B&H of symbols listed at split start, B&H@exp = that basket at the run's average exposure, rest cash.",
        "- Filters are assumed constant over history (tick/step per symbol in `research/g1_run.py`).",
    ]
    if projection:
        L += [f"- {projection}"]
    return "\n".join(L) + "\n"


def baseline_stats(bars, symbols, res_stat: Stat, start_cash: float) -> dict[str, dict]:
    eq = res_stat.daily_equity
    curves = mx.baseline_curves({s: bars[s] for s in symbols}, eq.index, start_cash,
                                res_stat.summary["avg_exposure"])
    return {
        k: {"total_return": mx.total_return(v), "max_drawdown": mx.max_drawdown(v)}
        for k, v in curves.items()
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--prereg", default="research/g1_preregistration.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--processes", type=int, default=0, help="0 = all cores")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke and Path(args.out).resolve() == Path(RESULTS_PATH).resolve():
        ap.error(f"--smoke must not write {RESULTS_PATH}")
    import os

    processes = args.processes or os.cpu_count() or 1
    t_start = time.time()
    prereg = json.loads(Path(args.prereg).read_text())
    shadow_syms = tuple(prereg["shadow_control"])
    symbols = tuple(s for s in args.symbols if s not in shadow_syms)
    if len(symbols) != len(args.symbols):
        print("note: shadow-control symbols are removed from the main universe", file=sys.stderr)
    trials = build_trials(prereg)
    splits = build_splits(prereg, args.smoke)
    if args.smoke:
        symbols = symbols[:2]
        trials = trials[:1]
    print(f"universe {symbols}; {len(trials)} trial(s); {len(splits)} splits; {processes} procs",
          file=sys.stderr)

    # data + features (once; workers inherit them via fork)
    horizon = prereg["fixed_grid"]["sigma_horizon_minutes"]
    need = symbols + shadow_syms
    bars: dict[str, pd.DataFrame] = {}
    for s in need:
        df = load_klines(s, Path(args.data_dir))
        if args.smoke:
            df = df.loc[splits[0].start - SMOKE_WARMUP : splits[-1].end]
        bars[s] = df
    t0 = time.time()
    feats = precompute_features(bars, horizon)
    print(f"  features: {time.time() - t0:.0f}s", file=sys.stderr)
    filters = {s: filters_for(s, bars[s]) for s in bars}
    shared = dict(bars=bars, features=feats, filters=filters)

    # 1. trials x splits (base)
    jobs: list[ReplayJob] = []
    for t in trials:
        jobs += make_jobs(prereg, t, splits, "base", symbols, "main")
    raw = run_set(jobs, splits, shared, processes, "trials x splits")
    stats = {(k[1], k[2]): v for k, v in raw.items()}

    # 2. selection (in-sample only)
    sel = select_trial(trials, stats, splits)
    sel_t = trials[sel]
    print(f"  selected {sel_t.label}", file=sys.stderr)

    # 3. stress on the selected trial (+ fees x2 is also the gate's CI input)
    jobs = []
    for v in ("fees2", "haircut"):
        jobs += make_jobs(prereg, sel_t, splits, v, symbols, "main")
    for v in ("base", "fees2"):
        jobs += make_jobs(prereg, sel_t, splits, v, shadow_syms, "shadow")
    raw = run_set(jobs, splits, shared, processes, "stress + shadow")
    stress = {(k[1], k[2], k[3]): v for k, v in raw.items() if k[0] == "main"}
    for sp in splits:
        stress[(sel, sp.name, "base")] = stats[(sel, sp.name)]
    shadow = {(k[2], k[3]): v for k, v in raw.items() if k[0] == "shadow"}

    gate = prereg["gate_g1"]
    main_base = {sp.name: stats[(sel, sp.name)] for sp in splits}
    main_f2 = {sp.name: stress[(sel, sp.name, "fees2")] for sp in splits}
    gate_rows = gate_g1(gate, splits, main_base, main_f2)
    shadow_gate = gate_g1(
        gate, splits,
        {sp.name: shadow[(sp.name, "base")] for sp in splits},
        {sp.name: shadow[(sp.name, "fees2")] for sp in splits},
    )

    # 5. shadow rule: ETH passing means the fill model may be too generous -> rerun at haircut 0.7
    rerun_rows = None
    if gate_passes(shadow_gate):
        print("  ETH control passed: rerunning the gate at fill_haircut 0.7", file=sys.stderr)
        jobs = make_jobs(prereg, sel_t, splits, "haircut_fees2", symbols, "main")
        raw = run_set(jobs, splits, shared, processes, "haircut rerun")
        hf2 = {k[2]: v for k, v in raw.items()}
        hb = {sp.name: stress[(sel, sp.name, "haircut")] for sp in splits}
        rerun_rows = gate_g1(gate, splits, hb, hf2)

    baselines = {
        sp.name: baseline_stats(bars, symbols, stats[(sel, sp.name)], prereg["start_cash"])
        for sp in splits
    }

    runtime = time.time() - t_start
    executed = (
        list(stats.values())
        + [v for k, v in stress.items() if k[2] != "base"]
        + list(shadow.values())
        + (list(raw.values()) if rerun_rows is not None else [])
    )
    day_of = {sp.name: sp.days for sp in splits}
    cpu_s = sum(x.elapsed for x in executed)
    done_days = sum(len(x.symbols) * day_of[x.split] for x in executed)
    per_day = cpu_s / done_days if done_days else float("nan")
    projection = f"Cost: {per_day:.3f} CPU-s per symbol-day over {len(executed)} runs ({cpu_s:.0f} CPU-s)."
    if args.smoke:
        full_splits = build_splits(prereg, False)
        full_bars = {
            s: pd.read_parquet(Path(args.data_dir) / "klines" / f"{s}_1m.parquet", columns=["close"])
            for s in need
        }
        for df in full_bars.values():
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        n_trials = len(build_trials(prereg))
        main_days = symbol_days(full_bars, full_splits, symbols)
        eth_days = symbol_days(full_bars, full_splits, shadow_syms)
        full_days = (n_trials + 2) * main_days + 2 * eth_days
        projection += (
            f" Projected full G1 on these {len(symbols)} symbols: {full_days:,.0f} symbol-days, "
            f"~{per_day * full_days / 3600:.1f} CPU-h, ~{per_day * full_days / processes / 3600:.1f} h on {processes} processes "
            "(smoke runs 2 symbols; per-symbol cost grows slightly with the universe)."
        )
    print("  " + projection, file=sys.stderr)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(
        prereg=prereg, smoke=args.smoke, trials=trials, splits=splits, symbols=symbols,
        stats=stats, sel=sel, stress=stress, shadow=shadow, shadow_gate=shadow_gate,
        gate_rows=gate_rows, rerun_rows=rerun_rows, baselines=baselines, runtime=runtime,
        projection=projection,
    ))
    print(f"wrote {path} ({runtime:.0f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
