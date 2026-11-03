"""G1 v2 runner: variants A (production DCA+TP), B (paired grid), C (rebalancing) under the same
FakeExchange, risk caps, halts, splits, stress, shadow control and gate as G1 v1.

Usage:
    python -m research.g1v2_run --out docs/research/G1V2_RESULTS.md          # the full run
    python -m research.g1v2_run --smoke --out /tmp/g1v2_smoke.md             # plumbing check

Everything comes from ``research/g1v2_preregistration.json`` (variants, trials, selection) and the
file it inherits, ``research/g1_preregistration.json`` (fees, risk, splits, stress, shadow, gate,
decision interval), plus the v1 addendum's gate interpretations. Neither file is modified here.

Flow:
  1. every trial of every variant x split, on a fresh shared-wallet portfolio
     (A: 10 trials re-run with the production ``decide``, v1 output files are never read;
      B: 3 trials; C: 3 trials; 16 in all);
  2. selection within each variant on in-sample splits only (highest median net Sharpe);
  3. each selected trial under the stress variants (fees x2, fill_haircut 0.7) on all splits;
  4. the ETHUSDT shadow control (own portfolio) for each selected trial;
  5. the inherited G1 gate, applied to each variant on its own (``research.g1_run.gate_g1``);
     if a variant's ETH control passes, its gate is re-run at fill_haircut 0.7;
  6. one markdown report with a side-by-side out-of-sample table and a gate table per variant.

B and C run through ``run_replay(strategy=...)``; the production ``GridParams`` handed to the
replay for them is only a carrier of ``sigma_horizon_minutes`` (the features), taken from the
v1 ``fixed_grid`` block, so all three variants see the same features.

``--smoke`` is a plumbing check (2 symbols, 1 trial per variant, Jan-Feb 2024, trimmed history):
not a G1 result. It also prints a projection of the full run's cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research import g1_run as g1
from vv.core.params import GridParams, RiskParams
from vv.core.variants.paired_grid import PairedGridParams, PairedGridStrategy
from vv.core.variants.rebalance import RebalanceParams, RebalanceStrategy
from vv.sim import metrics as mx
from vv.sim.replay import ReplayJob, precompute_features

RESULTS_PATH = "docs/research/G1V2_RESULTS.md"
PREREG_V1 = "research/g1_preregistration.json"
PREREG_V2 = "research/g1v2_preregistration.json"
A, B, C = "A_dca_tp", "B_paired_grid", "C_rebalance"
VARIANTS = (A, B, C)
SHORT = {A: "A DCA+TP", B: "B paired grid", C: "C rebalance"}
EXPECTED_TRIALS = 16


# --------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class VTrial:
    """A trial of variant B or C (variant A reuses ``g1_run.Trial``)."""

    variant: str
    idx: int
    value: float  # B: step, C: band
    label: str


def build_trials(v1: dict, v2: dict, smoke: bool) -> dict[str, list]:
    levels = v2["variants"][B]["fixed"]["levels"]
    out: dict[str, list] = {
        A: g1.build_trials(v1),
        B: [
            VTrial(B, i, x, f"B{i + 1:02d} step {x:.2%} x {levels} levels")
            for i, x in enumerate(v2["variants"][B]["trials"]["step"])
        ],
        C: [
            VTrial(C, i, x, f"C{i + 1:02d} band {x:.0%}")
            for i, x in enumerate(v2["variants"][C]["trials"]["band"])
        ],
    }
    if not smoke:
        n = sum(len(t) for t in out.values())
        if n != EXPECTED_TRIALS:
            raise SystemExit(f"pre-registration says {EXPECTED_TRIALS} trials in all, found {n}")
    else:
        out = {k: t[:1] for k, t in out.items()}
    return out


def carrier_grid(v1: dict) -> GridParams:
    """Production params handed to ``run_replay`` for B/C: only ``sigma_horizon_minutes`` is used."""
    g = v1["fixed_grid"]
    return GridParams(
        spacing_sigmas=1.5, min_step=0.01, sigma_horizon_minutes=g["sigma_horizon_minutes"],
        levels=g["levels"], qty_multiplier=g["qty_multiplier"],
        tp_fee_multiple=g["tp_fee_multiple"], maker_fee=g["maker_fee"],
    )


def make_jobs(
    v1: dict, v2: dict, variant: str, trial, splits: list[g1.Split], fee_variant: str,
    symbols: tuple[str, ...], tag: str,
) -> list[ReplayJob]:
    """Jobs keyed ``(tag, trial.idx, split, fee_variant)`` (the ``g1_run.run_set`` convention)."""
    if variant == A:
        return g1.make_jobs(v1, trial, splits, fee_variant, symbols, tag)
    mult, haircut = g1.VARIANTS[fee_variant]
    risk = RiskParams(**v1["risk"])
    if variant == B:
        levels = v2["variants"][B]["fixed"]["levels"]
        strategy = PairedGridStrategy(PairedGridParams(levels=levels, step=trial.value), risk)
    else:
        strategy = RebalanceStrategy(RebalanceParams(band=trial.value), risk)
    grid = carrier_grid(v1)
    fees = g1.fees_for(v1, mult)
    return [
        ReplayJob(
            key=(tag, trial.idx, sp.name, fee_variant),
            kwargs=dict(
                grid=grid, risk=risk, start=sp.start, end=sp.end,
                decision_minutes=v1["decision_interval_minutes"], start_cash=v1["start_cash"],
                fees=fees, fill_haircut=haircut, seed=g1.REPLAY_SEED, keep_fills=False,
                strategy=strategy,
            ),
            symbols=symbols,
        )
        for sp in sorted(splits, key=lambda s: -s.days)  # long jobs first
    ]


@dataclass
class VResult:
    variant: str
    trials: list
    stats: dict  # (trial idx, split) -> Stat, base costs
    sel: int
    stress: dict  # (split, "fees2" | "haircut") -> Stat; ("base") entries are in ``stats``
    shadow: dict  # (split, "base" | "fees2") -> Stat
    gate_rows: list[dict]
    shadow_gate: list[dict]
    rerun_rows: list[dict] | None
    baselines: dict  # split -> baseline_stats
    deployable: bool

    @property
    def trial(self):
        return self.trials[self.sel]


# --------------------------------------------------------------------------- rendering
def _gate_cells(rows: list[dict]) -> list[list[str]]:
    return [[r["name"], r["rule"], r["value"], "PASS" if r["passed"] else "FAIL"] for r in rows]


def render(
    *, v1, v2, smoke, splits, symbols, shadow_syms, res: dict[str, VResult], runtime, projection,
    n_trials,
) -> str:
    L: list[str] = []
    ins = [s.name for s in splits if s.role == "in_sample"]
    oos = [s for s in splits if s.role == "out_of_sample"]
    title = "G1 v2 results" + (" (SMOKE RUN, plumbing check only, not a G1 result)" if smoke else "")
    L += [f"# {title}", ""]
    counts = ", ".join(f"{SHORT[v]} {len(res[v].trials)}" for v in VARIANTS)
    L += [
        f"Universe: {', '.join(symbols)} (one shared wallet, {v1['start_cash']:,} USDT). "
        f"Pre-registration v{v2['version']} (inherits v1: fees, risk caps, halts, splits, stress, "
        f"shadow control, gate); decisions every {v1['decision_interval_minutes']} min. "
        f"**Total trials across variants = {n_trials}** ({counts}); every trial is reported below. "
        f"Runtime {runtime:.0f}s.",
        "",
    ]

    # ---- side by side OOS
    L += ["## Out-of-sample, side by side (selected trial per variant, base costs)", ""]
    rows = []
    for sp in oos:
        for v in VARIANTS:
            r = res[v]
            s = r.stats[(r.trial.idx, sp.name)]
            b = r.baselines[sp.name]
            rows.append([
                sp.name, f"{SHORT[v]}: {r.trial.label}", g1.pct(s.summary["total_return"]),
                f"{s.summary['sharpe']:.2f}", g1.pct(s.summary["max_drawdown"]),
                f"{s.summary['net_capture_over_fees']:.2f}", g1.pct(s.summary["avg_exposure"]),
                g1.pct(b["bh_scaled_to_exposure"]["total_return"]),
                g1.pct(b["equal_weight_bh"]["total_return"]),
            ])
    L += g1.md_table(
        ["split", "variant: selected trial", "return", "Sharpe", "MDD", "capture/fees", "avg exp",
         "B&H at same exposure", "B&H fully invested"],
        rows,
    )
    L += [
        "",
        "B&H = equal-weight buy-and-hold of the symbols listed at split start, no rebalancing; "
        "\"same exposure\" holds that basket at the variant's own average exposure, rest cash.",
        "",
    ]

    # ---- gate summary
    L += ["## Gate G1 per variant (each on its own)", ""]
    crit = [r["name"] for r in res[A].gate_rows]
    hdr = ["criterion", "rule"] + [SHORT[v] for v in VARIANTS]
    rows = []
    for i, name in enumerate(crit):
        row = [name, res[A].gate_rows[i]["rule"]]
        for v in VARIANTS:
            g = res[v].gate_rows[i]
            row.append(f"{'PASS' if g['passed'] else 'FAIL'} ({g['value']})")
        rows.append(row)
    rows.append(["ETH shadow control", "should FAIL"] + [
        "PASS: flag" if g1.gate_passes(res[v].shadow_gate) else "FAIL (expected)" for v in VARIANTS
    ])
    rows.append(["**Deployable**", "all criteria (+ haircut rerun if ETH passes)"] + [
        "**YES**" if res[v].deployable else "**NO**" for v in VARIANTS
    ])
    L += g1.md_table(hdr, rows)
    L += [""]

    # ---- per variant
    for v in VARIANTS:
        r = res[v]
        sel_t = r.trial
        L += [f"## {SHORT[v]}", ""]
        L += ["### Trials (in-sample only; selection = highest median net Sharpe)", ""]
        rows = []
        for t in r.trials:
            ss = [r.stats[(t.idx, n)] for n in ins]
            cap = mx.net_capture_over_fees(
                sum(x.summary["gross_pnl"] for x in ss), sum(x.summary["fees_paid"] for x in ss)
            )
            rows.append([
                t.label + (" **selected**" if t.idx == r.sel else ""),
                f"{np.median([x.summary['sharpe'] for x in ss]):.2f}",
                g1.pct(float(np.median([x.summary["total_return"] for x in ss]))),
                g1.pct(max(x.summary["max_drawdown"] for x in ss)),
                f"{sum(x.summary['round_trips'] for x in ss)}", f"{cap:.2f}",
            ])
        L += g1.md_table(
            ["trial", "median Sharpe", "median return", "worst MDD", "round trips", "capture/fees"], rows
        )

        L += ["", f"### Selected trial {sel_t.label}: all splits vs baselines (base costs)", ""]
        rows = []
        for sp in splits:
            s = r.stats[(sel_t.idx, sp.name)]
            b = r.baselines[sp.name]
            rows.append([
                f"{sp.name} ({'IS' if sp.role == 'in_sample' else 'OOS'})",
                g1.pct(s.summary["total_return"]), f"{s.summary['sharpe']:.2f}",
                g1.pct(s.summary["max_drawdown"]), f"{s.summary['net_capture_over_fees']:.2f}",
                g1.pct(s.summary["avg_exposure"]),
                g1.pct(b["equal_weight_bh"]["total_return"]),
                g1.pct(b["equal_weight_bh"]["max_drawdown"]),
                g1.pct(b["bh_scaled_to_exposure"]["total_return"]),
            ])
        L += g1.md_table(
            ["split", "return", "Sharpe", "MDD", "capture", "avg exp", "B&H ret", "B&H MDD",
             "B&H@exp ret"],
            rows,
        )

        L += ["", "### Stress (selected trial)", ""]
        rows = []
        for variant, label in (("fees2", "fees x2"), ("haircut", "fill_haircut 0.7")):
            for sp in splits:
                s = r.stress[(sp.name, variant)]
                rows.append([label, sp.name, g1.pct(s.summary["total_return"]),
                             f"{s.summary['sharpe']:.2f}", g1.pct(s.summary["max_drawdown"]),
                             f"{s.summary['net_capture_over_fees']:.2f}"])
        L += g1.md_table(["variant", "split", "return", "Sharpe", "MDD", "capture"], rows)

        L += ["", f"### Shadow control ({', '.join(shadow_syms)}, own portfolio, selected trial)", ""]
        rows = []
        for sp in splits:
            s = r.shadow[(sp.name, "base")]
            rows.append([sp.name, g1.pct(s.summary["total_return"]), f"{s.summary['sharpe']:.2f}",
                         g1.pct(s.summary["max_drawdown"]), f"{s.summary['net_capture_over_fees']:.2f}"])
        L += g1.md_table(["split", "return", "Sharpe", "MDD", "capture"], rows)
        eth_pass = g1.gate_passes(r.shadow_gate)
        L += ["", "ETH gate: " + (
            "PASS. **Flag: the fill model may be too generous.**" if eth_pass
            else "FAIL (the control behaves as expected)."), ""]
        L += g1.md_table(["criterion", "ETH value", "result"],
                         [[x["name"], x["value"], "PASS" if x["passed"] else "FAIL"] for x in r.shadow_gate])

        L += ["", "### Gate G1", ""]
        L += g1.md_table(["criterion", "rule", "value", "result"], _gate_cells(r.gate_rows))
        if r.rerun_rows is not None:
            L += ["", "Rerun at fill_haircut 0.7 (triggered by the ETH shadow rule):", ""]
            L += g1.md_table(["criterion", "rule", "value", "result"], _gate_cells(r.rerun_rows))
        failed = [x["name"] for x in r.gate_rows if not x["passed"]]
        if smoke:
            verdict = "Smoke run: not a verdict. The 2022 criterion cannot be met without the 2022 split."
        elif r.deployable:
            verdict = "PASS on the literal gate" + (
                " and on the haircut-0.7 rerun." if r.rerun_rows is not None else ".")
        elif failed:
            verdict = "FAIL. Criteria not met: " + ", ".join(failed) + ". Not deployable as pre-registered."
        else:
            verdict = ("The literal gate passes but the ETH control also passed and the haircut-0.7 "
                       "rerun does not: not trusted, not deployable.")
        L += ["", f"**Verdict ({SHORT[v]}).** {verdict}", ""]

    # ---- notes
    L += ["## Notes", ""]
    for v in VARIANTS:
        r = res[v]
        sel_all = [r.stats[(r.trial.idx, sp.name)] for sp in splits]
        rej: Counter[str] = Counter()
        for s in sel_all:
            rej.update(s.rejections)
        L += [
            f"- {SHORT[v]} (selected, base): {sum(s.n_places for s in sel_all)} placed, "
            f"{sum(s.n_cancels for s in sel_all)} cancelled over {sum(s.n_decisions for s in sel_all)} "
            f"decisions; halts {sum(s.halts for s in sel_all)}; peak exposure incl. resting buys "
            f"{g1.pct(max(s.max_exposure_total_post for s in sel_all))}; rejections {dict(rej) or 'none'}.",
        ]
    L += [
        f"- Trials run: {n_trials} x {len(splits)} splits = {n_trials * len(splits)} base runs; "
        "stress and shadow runs reuse each variant's selected trial. Variant A was re-run with the "
        "production strategy (no v1 output file is read).",
        "- Interpretations (v1 addendum, applied to every variant): capture = pooled gross realised / "
        "fees (all splits); drawdown = worst split; OOS CI = pooled OOS daily returns at fees x2; "
        "stress changes exchange fees only.",
        "- Open choices of B and C (documented in `vv/core/variants/paired_grid.py` and "
        "`rebalance.py`): B anchors at the mid when a symbol first warms up, quote per level = "
        "symbol budget / levels, sells are lifted to ask + 1 tick when price already passed the "
        "level; C needs warm features (sigma) like the others and marks positions at mid.",
        "- Filters are assumed constant over history (see `research/g1_run.py`).",
    ]
    if projection:
        L += [f"- {projection}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", help="default: the v2 pre-registered universe")
    ap.add_argument("--prereg", default=PREREG_V2)
    ap.add_argument("--prereg-v1", default=PREREG_V1)
    ap.add_argument("--out", help=f"default {RESULTS_PATH} (required with --smoke)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--processes", type=int, default=0, help="0 = all cores")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke and not args.out:
        ap.error("--smoke needs an explicit --out (it must not write " + RESULTS_PATH + ")")
    out_path = Path(args.out or RESULTS_PATH)
    if args.smoke and out_path.resolve() == Path(RESULTS_PATH).resolve():
        ap.error(f"--smoke must not write {RESULTS_PATH}")

    processes = args.processes or os.cpu_count() or 1
    t_start = time.time()
    v1 = json.loads(Path(args.prereg_v1).read_text())
    v2 = json.loads(Path(args.prereg).read_text())
    shadow_syms = tuple(v1["shadow_control"])
    symbols = tuple(s for s in (args.symbols or v2["universe"]) if s not in shadow_syms)
    splits = g1.build_splits(v1, args.smoke)
    trials = build_trials(v1, v2, args.smoke)
    n_trials = sum(len(t) for t in trials.values())
    if args.smoke:
        symbols = symbols[:2]
    print(f"universe {symbols}; trials { {v: len(t) for v, t in trials.items()} }; "
          f"{len(splits)} splits; {processes} procs", file=sys.stderr)

    horizon = v1["fixed_grid"]["sigma_horizon_minutes"]
    need = symbols + shadow_syms
    bars: dict[str, pd.DataFrame] = {}
    for s in need:
        df = g1.load_klines(s, Path(args.data_dir))
        if args.smoke:
            df = df.loc[splits[0].start - g1.SMOKE_WARMUP : splits[-1].end]
        bars[s] = df
    t0 = time.time()
    feats = precompute_features(bars, horizon)
    print(f"  features: {time.time() - t0:.0f}s", file=sys.stderr)
    filters = {s: g1.filters_for(s, bars[s]) for s in bars}
    shared = dict(bars=bars, features=feats, filters=filters)

    # 1. every trial x split, base costs (A first: it is the slowest)
    jobs: list[ReplayJob] = []
    for v in VARIANTS:
        for t in trials[v]:
            jobs += make_jobs(v1, v2, v, t, splits, "base", symbols, f"{v}|main")
    raw = g1.run_set(jobs, splits, shared, processes, "trials x splits")
    stats: dict[str, dict] = {v: {} for v in VARIANTS}
    for k, st in raw.items():
        stats[k[0].split("|")[0]][(k[1], k[2])] = st

    # 2. selection within each variant (in-sample only)
    sel = {v: g1.select_trial(trials[v], stats[v], splits) for v in VARIANTS}
    for v in VARIANTS:
        print(f"  {SHORT[v]}: selected {trials[v][sel[v]].label}", file=sys.stderr)

    # 3 + 4. stress and shadow for each selected trial
    jobs = []
    for v in VARIANTS:
        t = trials[v][sel[v]]
        for fv in ("fees2", "haircut"):
            jobs += make_jobs(v1, v2, v, t, splits, fv, symbols, f"{v}|main")
        for fv in ("base", "fees2"):
            jobs += make_jobs(v1, v2, v, t, splits, fv, shadow_syms, f"{v}|shadow")
    raw2 = g1.run_set(jobs, splits, shared, processes, "stress + shadow")

    # 5. gate per variant (+ haircut rerun where the ETH control passes)
    gate = v1["gate_g1"]
    results: dict[str, VResult] = {}
    rerun_stats: list = []
    for v in VARIANTS:
        t = trials[v][sel[v]]
        stress = {(k[2], k[3]): x for k, x in raw2.items() if k[0] == f"{v}|main"}
        shadow = {(k[2], k[3]): x for k, x in raw2.items() if k[0] == f"{v}|shadow"}
        base = {sp.name: stats[v][(t.idx, sp.name)] for sp in splits}
        fees2 = {sp.name: stress[(sp.name, "fees2")] for sp in splits}
        gate_rows = g1.gate_g1(gate, splits, base, fees2)
        shadow_gate = g1.gate_g1(
            gate, splits, {sp.name: shadow[(sp.name, "base")] for sp in splits},
            {sp.name: shadow[(sp.name, "fees2")] for sp in splits},
        )
        rerun_rows = None
        if g1.gate_passes(shadow_gate):
            print(f"  {SHORT[v]}: ETH control passed, rerunning the gate at fill_haircut 0.7",
                  file=sys.stderr)
            rj = make_jobs(v1, v2, v, t, splits, "haircut_fees2", symbols, f"{v}|main")
            rr = g1.run_set(rj, splits, shared, processes, f"{SHORT[v]} haircut rerun")
            rerun_stats += list(rr.values())
            hb = {sp.name: stress[(sp.name, "haircut")] for sp in splits}
            rerun_rows = g1.gate_g1(gate, splits, hb, {k[2]: x for k, x in rr.items()})
        deployable = g1.gate_passes(gate_rows) and (
            rerun_rows is None or g1.gate_passes(rerun_rows)
        )
        baselines = {
            sp.name: g1.baseline_stats(bars, symbols, base[sp.name], v1["start_cash"]) for sp in splits
        }
        results[v] = VResult(v, trials[v], stats[v], sel[v], stress, shadow, gate_rows,
                             shadow_gate, rerun_rows, baselines, bool(deployable))

    runtime = time.time() - t_start
    full_symbols = tuple(s for s in (args.symbols or v2["universe"]) if s not in shadow_syms)
    projection = project_cost(args, v1, results, splits, full_symbols, shadow_syms, processes,
                              raw, raw2, rerun_stats)
    print("  " + projection, file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(
        v1=v1, v2=v2, smoke=args.smoke, splits=splits, symbols=symbols, shadow_syms=shadow_syms,
        res=results, runtime=runtime, projection=projection, n_trials=n_trials,
    ))
    print(f"wrote {out_path} ({runtime:.0f}s)", file=sys.stderr)
    return 0


def project_cost(args, v1, results, splits, full_symbols, shadow_syms, processes, raw, raw2,
                 rerun_stats) -> str:
    """Executed cost per symbol-day by variant (wall seconds inside ``run_replay``) and, in smoke
    mode, a projection of the full run on the full history."""
    day_of = {sp.name: sp.days for sp in splits}
    per_day: dict[str, float] = {}
    cpu = 0.0
    for v in VARIANTS:
        ex = [x for k, x in raw.items() if k[0].startswith(v)]
        ex += [x for k, x in raw2.items() if k[0].startswith(v)]
        c = sum(x.elapsed for x in ex)
        d = sum(len(x.symbols) * day_of[x.split] for x in ex)
        per_day[v] = c / d if d else float("nan")
        cpu += c
    cpu += sum(x.elapsed for x in rerun_stats)
    txt = "Cost per symbol-day: " + ", ".join(f"{SHORT[v]} {per_day[v]:.3f}s" for v in VARIANTS) + (
        f" ({cpu:.0f} CPU-s executed; wall time inside run_replay, so inflated if the machine "
        "was busy)."
    )
    if not args.smoke:
        return txt
    full_splits = g1.build_splits(v1, False)
    full = {
        s: pd.read_parquet(Path(args.data_dir) / "klines" / f"{s}_1m.parquet", columns=["close"])
        for s in full_symbols + shadow_syms
    }
    for df in full.values():
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    main_days = g1.symbol_days(full, full_splits, full_symbols)
    eth_days = g1.symbol_days(full, full_splits, shadow_syms)
    n_full = {A: 10, B: 3, C: 3}
    total = 0.0
    parts = []
    for v in VARIANTS:
        days = (n_full[v] + 2) * main_days + 2 * eth_days
        total += per_day[v] * days
        parts.append(f"{SHORT[v]} {per_day[v] * days / 3600:.1f} CPU-h")
    return txt + (
        f" Projected full v2 on {len(full_symbols)} symbols + ETH shadow: {', '.join(parts)}; total ~{total / 3600:.1f} "
        f"CPU-h, ~{total / processes / 3600:.1f} h on {processes} processes (before any haircut "
        "reruns; the smoke measured 2 symbols, per-symbol cost grows slightly with the universe)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
