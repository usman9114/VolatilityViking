"""G1 diagnostics (WP-S1.7): exposure audit, fill-depth sweep, VR attribution, cash benchmark.

These are diagnostics, not trials: they change no gate and no pre-registered setting. Everything
runs the selected v1 trial T09 (min_step 0.03, spacing_sigmas 1.5) from
``research/g1_preregistration.json`` unchanged.

Usage:
    python -m research.g1_diagnostics --symbols BTCUSDT SOLUSDT XRPUSDT DOGEUSDT LINKUSDT \
        --processes 1 --out docs/research/G1_DIAGNOSTICS.md
    python -m research.g1_diagnostics --symbols ... --reuse-raw   # re-render from saved numbers

Sections:
  1. Exposure audit: T09 on 2021, rebuilt from a per-minute trace of the exchange state
     (``TracingExchange``) and cross-checked against the fills and the replay's decision rows.
  2. Fill-depth sweep: T09 on every v1 split at ``min_pierce`` in {0, 0.0005, 0.0015}.
  3. VR attribution: single-symbol replays (one wallet per coin) per calendar year, regressed
     on that coin-year's 1h VR from ``docs/research/G0_RESULTS.md``.
  4. Cash benchmark: 4%/yr USDT yield next to T09's return per split.

``vv/sim/replay.py`` is not touched: ``min_pierce`` and the tracing exchange are threaded into
``run_replay`` by swapping ``vv.sim.replay.FakeExchange`` for the duration of a ``with`` block
(``pierce_exchange``). That works for ``--processes 1`` and for forked workers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import vv.sim.replay as replay_mod
from research import g1_run as g1
from vv.core.params import RiskParams
from vv.sim import metrics as mx
from vv.sim.fake_exchange import BAR, FakeExchange
from vv.sim.replay import ReplayJob, precompute_features, run_many

RAW_PATH = "data/g1_diagnostics_raw.json"  # data/ is gitignored
OUT_PATH = "docs/research/G1_DIAGNOSTICS.md"
G0_PATH = "docs/research/G0_RESULTS.md"
PIERCES = (0.0, 0.0005, 0.0015)
T09 = (0.03, 1.5)  # (min_step, spacing_sigmas) in the v1 pre-registration
AUDIT_SPLIT = "2021"
VR_COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT")
TOP_DAYS = 10


# --------------------------------------------------------------------------- exchange hooks
class TracingExchange(FakeExchange):
    """``FakeExchange`` that samples its own state after matching every 1m bar.

    One sample per bar close: holdings at that bar's close, the notional of resting buys
    (``price * qty``, the quantity ``exposure`` counts) and total equity. Orders only change at
    decision times and on fills, so per-minute samples capture the state exactly. Behaviour is
    unchanged (matching is the base class's ``_match_bar``).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_ns: list[int] = []
        self.trace_held: list[float] = []
        self.trace_resting: list[float] = []
        self.trace_equity: list[float] = []

    def _on_advance(self, old, new) -> None:  # type: ignore[override]
        for k in range(self._n_closed(old), self._n_closed(new)):
            for symbol in self._bars:
                self._match_bar(symbol, k)
            self._sample(k)

    def _sample(self, k: int) -> None:
        held = 0.0
        for s, p in self._pos.items():
            if p.qty > 0:
                held += p.qty * self._close_ff[s][k]
        resting = reserved = 0.0
        for o in self._orders.values():
            if o.side == "buy":
                resting += o.price * o.qty
                reserved += o.reserved
        self.trace_ns.append(int(self._index.asi8[k]) + BAR.value)
        self.trace_held.append(held)
        self.trace_resting.append(resting)
        self.trace_equity.append(self.cash_free + reserved + held)


@contextlib.contextmanager
def pierce_exchange(min_pierce: float = 0.0, trace: bool = False) -> Iterator[list[TracingExchange]]:
    """Make ``run_replay`` build its exchange with ``min_pierce`` (and optionally tracing).

    Yields the list that collects every ``TracingExchange`` built inside the block. The
    original ``FakeExchange`` is restored on exit.
    """
    orig = replay_mod.FakeExchange
    cls = TracingExchange if trace else orig
    made: list[TracingExchange] = []

    def factory(*args: Any, **kwargs: Any):
        kwargs.setdefault("min_pierce", min_pierce)
        ex = cls(*args, **kwargs)
        if trace:
            made.append(ex)  # type: ignore[arg-type]
        return ex

    replay_mod.FakeExchange = factory  # type: ignore[assignment]
    try:
        yield made
    finally:
        replay_mod.FakeExchange = orig


# --------------------------------------------------------------------------- statistics
def ols_multi(X: Sequence[Sequence[float]] | np.ndarray, y: Sequence[float]) -> dict[str, Any]:
    """OLS of ``y`` on the columns of ``X`` plus an intercept (first coefficient)."""
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    ya = np.asarray(y, dtype=float)
    n, k = Xa.shape
    D = np.column_stack([np.ones(n), Xa])
    beta, *_ = np.linalg.lstsq(D, ya, rcond=None)
    resid = ya - D @ beta
    dof = n - D.shape[1]
    sse = float(resid @ resid)
    sst = float(((ya - ya.mean()) ** 2).sum())
    if dof > 0:
        cov = (sse / dof) * np.linalg.inv(D.T @ D)
        se = np.sqrt(np.diag(cov))
        t = beta / se
    else:
        t = np.full(D.shape[1], float("nan"))
    return {
        "coef": beta, "t": t, "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "n": n, "resid": resid, "dof": dof,
    }


def ols(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Simple regression ``y = intercept + slope * x``: slope, its t-stat, R^2, residuals."""
    r = ols_multi(np.asarray(x, dtype=float)[:, None], y)
    return {
        "slope": float(r["coef"][1]), "intercept": float(r["coef"][0]),
        "t_slope": float(r["t"][1]), "r2": float(r["r2"]), "n": r["n"], "resid": r["resid"],
    }


def parse_yearly_vr(text: str, robust: bool = False) -> dict[tuple[str, int], float]:
    """1h VR per (symbol, year) from the G0 yearly stability table (baseline or robust)."""
    head = "## Stability: per calendar year, robust" if robust else "## Stability: per calendar year (VR"
    start = text.index(head)
    nxt = text.find("\n## ", start + 1)
    block = text[start: nxt if nxt != -1 else len(text)]
    out: dict[tuple[str, int], float] = {}
    for line in block.splitlines():
        m = re.match(r"\|\s*([A-Z0-9]+USDT)\s*\|\s*(\d{4})\s*\|\s*([0-9.]+)\s*\|", line)
        if m:
            out[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return out


# --------------------------------------------------------------------------- section 1
def _sample_frame(tr: TracingExchange, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.DatetimeIndex(np.asarray(tr.trace_ns, dtype="int64"), tz="UTC", name="ts")
    df = pd.DataFrame(
        {"held": tr.trace_held, "resting": tr.trace_resting, "equity": tr.trace_equity}, index=idx
    )
    return df.loc[start:end]


def holdings_from_fills(
    fills: list[dict], bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex
) -> pd.Series:
    """Holdings value at each sample time, from fills and raw 1m closes only (no exchange state)."""
    held = np.zeros(len(index))
    for s in {f["symbol"] for f in fills}:
        fs = [f for f in fills if f["symbol"] == s]
        ts = pd.DatetimeIndex([f["ts"] for f in fs]).tz_convert("UTC")
        dq = pd.Series([f["qty"] if f["side"] == "buy" else -f["qty"] for f in fs], index=ts)
        dq = dq.groupby(level=0).sum().sort_index().cumsum()
        qty = dq.reindex(dq.index.union(index)).ffill().reindex(index).fillna(0.0)
        close = bars[s]["close"].reindex(index - BAR, method="ffill")
        held += qty.to_numpy() * close.to_numpy()
    return pd.Series(held, index=index)


def run_audit(
    prereg: dict, trial: g1.Trial, split: g1.Split, symbols: tuple[str, ...], shared: dict,
) -> dict[str, Any]:
    risk = RiskParams(**prereg["risk"])
    grid = g1.grid_for(prereg, trial)
    with pierce_exchange(0.0, trace=True) as made:
        res = replay_mod.run_replay(
            bars={s: shared["bars"][s] for s in symbols},
            features={s: shared["features"][s] for s in symbols},
            filters=shared["filters"], grid=grid, risk=risk, start=split.start, end=split.end,
            decision_minutes=prereg["decision_interval_minutes"], start_cash=prereg["start_cash"],
            fees=g1.fees_for(prereg, 1.0), fill_haircut=1.0, seed=g1.REPLAY_SEED,
            record_trace=True, keep_fills=True,
        )
    tr = made[0]
    fr = _sample_frame(tr, split.start, split.end)
    a = mx.exposure_audit(fr["held"], fr["resting"], fr["equity"])

    # cross-checks, independent of the exchange's own bookkeeping
    fills_held = holdings_from_fills(res.fills, {s: shared["bars"][s] for s in symbols}, fr.index)
    held_gap = float((fills_held - fr["held"]).abs().max())
    dec = res.decisions
    at_dec = fr.reindex(dec.index)  # NaN where no bar closed at that minute (data gaps)
    eq_gap = float((at_dec["equity"] - dec["equity"]).abs().max())
    held_dec_gap = float((at_dec["held"] - dec["exposure"] * dec["equity"]).abs().max())
    no_sample = int(at_dec["equity"].isna().sum())
    return {
        "split": split.name, "n_samples": len(fr), "n_decisions": int(len(dec)),
        "decisions_without_sample": no_sample,
        "reported_avg_exposure": float(res.avg_exposure),
        "reported_raw_mean": float(dec["exposure"].mean()),
        "decision_total_post": float(dec["exposure_total_post"].mean()),
        "decision_deployed": mx.deployed_from_decisions(dec),
        "decision_max_total_post": float(dec["exposure_total_post"].max()),
        "halt_share_of_decisions": float(dec["halt_active"].mean()),
        "time_in_market": float(res.time_in_market),
        "final_return": float(res.final_equity / res.start_cash - 1.0),
        "final_equity": float(res.final_equity),
        "n_round_trips": int(res.round_trips), "n_fills": int(res.n_fills),
        "audit": a,
        "check_held_vs_fills_max_abs_usdt": held_gap,
        "check_equity_vs_decisions_max_abs_usdt": eq_gap,
        "check_held_vs_decisions_max_abs_usdt": held_dec_gap,
        "trace_len": len(res.trace or []),
    }


# --------------------------------------------------------------------------- section 2
def run_sweep(
    prereg: dict, trial: g1.Trial, splits: list[g1.Split], symbols: tuple[str, ...], shared: dict,
    processes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mp in PIERCES:
        jobs = g1.make_jobs(prereg, trial, splits, "base", symbols, "main")
        with pierce_exchange(mp):
            stats = g1.run_set(jobs, splits, shared, processes, f"min_pierce {mp:g}")
        for sp in splits:
            st = stats[("main", trial.idx, sp.name, "base")]
            s = st.summary
            pnl = mx.daily_pnl(st.daily_equity)
            rows.append({
                "min_pierce": mp, "split": sp.name, "role": sp.role,
                "total_return": s["total_return"], "sharpe": s["sharpe"],
                "capture": s["net_capture_over_fees"], "round_trips": s["round_trips"],
                "fees_paid": s["fees_paid"], "gross_pnl": s["gross_pnl"],
                "top10_share": mx.top_n_day_pnl_share(pnl, TOP_DAYS),
                "cash_yield_return": s["cash_yield_return"],
                "excess_over_cash": s["excess_over_cash"],
                "max_drawdown": s["max_drawdown"], "avg_exposure": s["avg_exposure"],
                "n_days": int(len(pnl)),
            })
    return rows


# --------------------------------------------------------------------------- section 3
def year_windows(years: Sequence[int]) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    out = {}
    for y in years:
        end = g1._utc(f"{y + 1}-01-01")
        if y == 2026:
            end = g1._utc("2026-11-01")  # same horizon as the v1 splits
        out[y] = (g1._utc(f"{y}-01-01"), end)
    return out


def run_attribution(
    prereg: dict, trial: g1.Trial, shared: dict, vr: dict[tuple[str, int], float],
    vr_robust: dict[tuple[str, int], float], processes: int,
) -> list[dict[str, Any]]:
    risk = RiskParams(**prereg["risk"])
    grid = g1.grid_for(prereg, trial)
    fees = g1.fees_for(prereg, 1.0)
    cells = sorted((s, y) for (s, y) in vr if s in VR_COINS)
    win = year_windows(sorted({y for _, y in cells}))
    jobs = []
    for s, y in cells:
        # A lone symbol's exchange clock starts at its first bar, so a window that opens before
        # the listing (LINK 2019, DOGE 2019, SOL 2020) starts one pre-roll hour after it. The
        # strategy is not warm that early anyway, so nothing is skipped.
        start = max(win[y][0], shared["bars"][s].index[0] + replay_mod._PREROLL)
        jobs.append(ReplayJob(
            key=(s, y),
            kwargs=dict(
                grid=grid, risk=risk, start=start, end=win[y][1],
                decision_minutes=prereg["decision_interval_minutes"],
                start_cash=prereg["start_cash"], fees=fees, fill_haircut=1.0,
                seed=g1.REPLAY_SEED, keep_fills=False,
            ),
            symbols=(s,),
        ))
    t0 = time.time()
    res = run_many(jobs, processes=processes, **shared)
    print(f"  attribution: {len(jobs)} single-symbol runs in {time.time() - t0:.0f}s", file=sys.stderr)
    rows = []
    for s, y in cells:
        r = res[(s, y)]
        rows.append({
            "symbol": s, "year": y, "vr": vr[(s, y)], "vr_robust": vr_robust.get((s, y)),
            "pnl": r.final_equity - r.start_cash, "gross_pnl": r.gross_pnl,
            "fees": r.fees_paid, "avg_exposure": r.avg_exposure, "round_trips": r.round_trips,
        })
    return rows


def attribution_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Regressions of coin-year P&L on 1h VR, plus the 'capture decay' tests."""
    y = [r["pnl"] for r in rows]
    vr = [r["vr"] for r in rows]
    out: dict[str, Any] = {"n": len(rows)}

    def pack(o: dict[str, Any]) -> dict[str, float]:
        return {k: float(o[k]) for k in ("slope", "intercept", "t_slope", "r2")} | {"n": int(o["n"])}

    base = ols(vr, y)
    out["base"] = pack(base)
    out["x_intercept"] = (-base["intercept"] / base["slope"]) if base["slope"] else float("nan")
    rob = [r for r in rows if r["vr_robust"] is not None]
    out["robust"] = pack(ols([r["vr_robust"] for r in rob], [r["pnl"] for r in rob]))
    out["no_doge_2019_2020"] = pack(ols(
        [r["vr"] for r in rows if not (r["symbol"] == "DOGEUSDT" and r["year"] <= 2020)],
        [r["pnl"] for r in rows if not (r["symbol"] == "DOGEUSDT" and r["year"] <= 2020)],
    ))
    # does time add anything once VR is known? (capture decay = negative year term)
    yrs = np.array([r["year"] - 2019 for r in rows], dtype=float)
    two = ols_multi(np.column_stack([vr, yrs]), y)
    out["with_year"] = {
        "vr_coef": float(two["coef"][1]), "vr_t": float(two["t"][1]),
        "year_coef": float(two["coef"][2]), "year_t": float(two["t"][2]), "r2": float(two["r2"]),
    }
    resid = base["resid"]
    early = [i for i, r in enumerate(rows) if r["year"] <= 2021]
    late = [i for i, r in enumerate(rows) if r["year"] >= 2024]
    out["resid_early_mean"] = float(np.mean(resid[early])) if early else float("nan")
    out["resid_late_mean"] = float(np.mean(resid[late])) if late else float("nan")
    out["n_early"], out["n_late"] = len(early), len(late)
    out["mean_pnl_early"] = float(np.mean([y[i] for i in early])) if early else float("nan")
    out["mean_pnl_late"] = float(np.mean([y[i] for i in late])) if late else float("nan")
    out["mean_vr_early"] = float(np.mean([vr[i] for i in early])) if early else float("nan")
    out["mean_vr_late"] = float(np.mean([vr[i] for i in late])) if late else float("nan")
    # rank correlation (robust to the few huge 2021 cells) and the early -> late decomposition:
    # change in mean P&L = slope * change in mean VR (regime) + change in mean residual (capture)
    out["spearman"] = float(pd.Series(vr).corr(pd.Series(y), method="spearman"))
    out["delta_pnl"] = out["mean_pnl_late"] - out["mean_pnl_early"]
    out["delta_from_vr"] = base["slope"] * (out["mean_vr_late"] - out["mean_vr_early"])
    out["delta_from_resid"] = out["resid_late_mean"] - out["resid_early_mean"]
    return out


# --------------------------------------------------------------------------- rendering
def _p(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def _f(x: float, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def render(raw: dict[str, Any]) -> str:
    L: list[str] = []
    au, sw, at, ast = raw["audit"], raw["sweep"], raw["attribution"], raw["attribution_stats"]
    a = au["audit"]
    L += [
        "# G1 diagnostics (WP-S1.7)",
        "",
        f"Selected v1 trial T09 (min_step 0.03, spacing 1.5 sigma), {', '.join(raw['symbols'])}, "
        "10,000 USDT, base costs. These are diagnostics, not trials: no gate or pre-registered "
        "setting changed, and the v1 FAIL stands.",
        "",
    ]

    # 1 ----------------------------------------------------------------------------
    L += ["## 1. Exposure audit (2021)", ""]
    L += [raw["conclusions"][0], ""]
    L += g1.md_table(
        ["measure (2021, T09)", "value"],
        [
            ["reported `avg_exposure` (decision-sampled, holdings, fraction of equity)",
             _p(au["reported_avg_exposure"], 2)],
            ["holdings only, time-weighted per minute", _p(a["holdings_exposure"], 2)],
            ["holdings + resting buys, time-weighted per minute", _p(a["total_exposure"], 2)],
            ["holdings + resting buys, sampled after each decision (`exposure_total_post`)",
             _p(au["decision_total_post"], 2)],
            ["average held / deployed (USDT)",
             f"{a['avg_held_usdt']:,.0f} / {a['avg_deployed_usdt']:,.0f}"],
            ["net P&L (USDT) and return", f"{a['pnl']:,.0f} ({_p(au['final_return'])})"],
            ["return on average held capital", _p(a["return_on_avg_held"])],
            ["**return on average deployed capital** (held + resting)",
             f"**{_p(a['return_on_avg_deployed'])}**"],
        ],
    )
    L += [
        "",
        "`avg_exposure` is a fraction of equity (0.0112, shown as 1.1%): mean over decisions of "
        "holdings at mark, before that decision's orders, **without resting buys**. The arithmetic is "
        "right, so it stays; additive fields `avg_total_exposure`, `avg_deployed_usdt` and "
        "`return_on_avg_deployed` (`summarize`, `deployed_from_decisions`) now cover holdings plus "
        "resting buys, the figure the 38% cap bounds.",
        "",
        f"Checks: holdings rebuilt from the fills alone match the exchange trace to "
        f"{au['check_held_vs_fills_max_abs_usdt']:.0e} USDT and the replay's decision rows to "
        f"{au['check_held_vs_decisions_max_abs_usdt']:.0e} ({au['decisions_without_sample']} decisions "
        "in no-bar minutes skipped); the traced run "
        f"{'equals' if au['traced_equals_plain'] else 'DIFFERS FROM'} the plain run. Halted on "
        f"{_p(au['halt_share_of_decisions'], 0)} of decisions. `record_trace=True` only records "
        "which feature bar each decision used, so exposure is rebuilt from the exchange state.",
        "",
    ]

    # 2 ----------------------------------------------------------------------------
    L += ["## 2. Fill-depth sweep", "", raw["conclusions"][1], ""]
    by = {(r["split"], r["min_pierce"]): r for r in sw}
    splits = [s for s in dict.fromkeys(r["split"] for r in sw)]
    hdr = ["split"] + [
        f"pierce {mp:g}" + (" (1 tick)" if mp == 0 else "") for mp in PIERCES
    ]
    rows = []
    for s in splits:
        cells = []
        for mp in PIERCES:
            r = by[(s, mp)]
            cells.append(
                f"{_p(r['total_return'])} / {_f(r['sharpe'])} / {_f(r['capture'], 1)} / "
                f"{_p(r['top10_share'], 0)}"
            )
        rows.append([f"{s} ({'IS' if by[(s, 0.0)]['role'] == 'in_sample' else 'OOS'})"] + cells)
    L += g1.md_table(hdr, rows)
    L += [
        "",
        "Cells: return / Sharpe / capture (gross P&L over fees) / share of total P&L earned on the "
        f"{TOP_DAYS} best days (daily equity changes; can pass 100% when other days lose; n/a if "
        "the split lost money). Pierce 0 reproduces the v1 table.",
        "",
    ]

    # 3 ----------------------------------------------------------------------------
    b, wy = ast["base"], ast["with_year"]
    L += ["## 3. VR attribution", "", raw["conclusions"][2], ""]
    L += [
        f"One wallet per coin, T09 params, calendar years, {ast['n']} coin-years "
        f"({', '.join(c.replace('USDT', '') for c in VR_COINS)}). P&L (USDT) on 1h VR "
        "(G0 baseline yearly table):",
        "",
    ]
    L += g1.md_table(
        ["regression", "slope per 1.0 VR", "t", "R2", "n"],
        [
            ["P&L ~ VR (all)", f"{b['slope']:,.0f}", _f(b["t_slope"], 1), _f(b["r2"]), str(b["n"])],
            ["P&L ~ VR, without DOGE 2019-20", f"{ast['no_doge_2019_2020']['slope']:,.0f}",
             _f(ast["no_doge_2019_2020"]["t_slope"], 1), _f(ast["no_doge_2019_2020"]["r2"]),
             str(ast["no_doge_2019_2020"]["n"])],
        ],
    )
    L += [
        "",
        f"Spearman {ast['spearman']:+.2f}; robust-VR table slope {ast['robust']['slope']:,.0f} "
        f"(t {_f(ast['robust']['t_slope'], 1)}). Mean VR rose from {_f(ast['mean_vr_early'], 3)} (2019-21, "
        f"n={ast['n_early']}) to {_f(ast['mean_vr_late'], 3)} (2024-26, n={ast['n_late']}) while mean P&L "
        f"fell from {ast['mean_pnl_early']:+,.0f} to {ast['mean_pnl_late']:+,.0f} USDT: the VR line "
        f"explains {ast['delta_from_vr']:+,.0f} of that {ast['delta_pnl']:+,.0f}, the rest "
        f"({ast['delta_from_resid']:+,.0f}) is lower P&L at the same VR (year term "
        f"t = {_f(wy['year_t'], 1)}; VR t = {_f(wy['vr_t'], 1)} once year is in). 2021 (DOGE, SOL) "
        "alone lifts the early mean and n is small: borderline evidence.",
        "",
        "Scatter (cells are `VR / P&L USDT`; blank = no VR row):",
        "",
    ]
    years = sorted({r["year"] for r in at})
    cell = {(r["symbol"], r["year"]): f"{r['vr']:.2f} / {r['pnl']:+,.0f}" for r in at}
    L += g1.md_table(
        ["coin"] + [str(y) for y in years],
        [[c.replace("USDT", "")] + [cell.get((c, y), "") for y in years] for c in VR_COINS],
    )
    L += [""]

    # 4 ----------------------------------------------------------------------------
    L += ["## 4. Cash benchmark (4% a year in USDT)", "", raw["conclusions"][3], ""]
    rows = []
    for s in splits:
        r = by[(s, 0.0)]
        rows.append([
            f"{s} ({'IS' if r['role'] == 'in_sample' else 'OOS'})", _p(r["total_return"]),
            _p(r["cash_yield_return"]), f"{r['excess_over_cash'] * 100:+.1f} pp",
            "yes" if r["excess_over_cash"] > 0 else "no",
        ])
    L += g1.md_table(["split", "T09 return", "4% cash", "excess", "beats cash"], rows)
    L += [
        "",
        f"Runtime {raw['runtime']:.0f}s on 1 process. `cash_yield_return` and `excess_over_cash` are "
        "now in `vv.sim.metrics.summarize`.",
    ]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- conclusions
def conclusions(raw: dict[str, Any]) -> list[str]:
    """One plain-language line per section, written from the numbers."""
    au, sw, ast = raw["audit"], raw["sweep"], raw["attribution_stats"]
    a = au["audit"]
    by = {(r["split"], r["min_pierce"]): r for r in sw}
    ok = max(au["check_held_vs_fills_max_abs_usdt"], au["check_equity_vs_decisions_max_abs_usdt"],
             au["check_held_vs_decisions_max_abs_usdt"])
    rep, hold, tot = au["reported_avg_exposure"], a["holdings_exposure"], a["total_exposure"]
    metric_ok = abs(rep - hold) <= 0.25 * hold + 1e-4
    c1 = (
        f"**Conclusion: the reported {_p(rep)} is a correct holdings-only number "
        f"(trace says {_p(hold)}) but not the capital the grid ties up: with resting buys "
        f"exposure is {_p(tot)}, so the 2021 return is {_p(a['return_on_avg_deployed'])} on average "
        f"deployed capital, not {_p(a['return_on_avg_held'])} on held capital.**"
        if metric_ok and ok < 1e-3 else
        f"**Conclusion: the reported {_p(rep)} does not match the per-minute trace ({_p(hold)} "
        f"holdings, {_p(tot)} with resting buys); the corrected return on deployed capital is "
        f"{_p(a['return_on_avg_deployed'])}.**"
    )
    base_split = [s for s in dict.fromkeys(r["split"] for r in sw)]
    r_all = {mp: [by[(s, mp)]["total_return"] for s in base_split] for mp in PIERCES}
    sh_all = {mp: [by[(s, mp)]["sharpe"] for s in base_split] for mp in PIERCES}
    win0 = [by[(s, 0.0)] for s in base_split if by[(s, 0.0)]["total_return"] > 0]
    t10 = [r["top10_share"] for r in win0]
    n_pos = {mp: sum(1 for x in r_all[mp] if x > 0) for mp in PIERCES}
    drop = {
        s: (by[(s, 0.0)]["total_return"], by[(s, 0.0005)]["total_return"]) for s in base_split
    }
    worst = min(drop, key=lambda k: drop[k][1] - drop[k][0])
    kept = [s for s, (r0, r1) in drop.items() if r0 > 0 and r1 >= 0.75 * r0]
    c2 = (
        f"**Conclusion: a 0.05% fill-depth requirement cuts the median split return from "
        f"{_p(float(np.median(r_all[0.0])))} to {_p(float(np.median(r_all[0.0005])))} ("
        f"{_p(float(np.median(r_all[0.0015])))} at 0.15%; median Sharpe "
        f"{_f(float(np.median(sh_all[0.0])))} / {_f(float(np.median(sh_all[0.0005])))} / "
        f"{_f(float(np.median(sh_all[0.0015])))}), hitting {worst} hardest "
        f"({_p(drop[worst][0])} to {_p(drop[worst][1])}); {', '.join(kept) or 'no split'} keep "
        f"at least 75% of their return; positive splits {n_pos[0.0]}/{n_pos[0.0005]}/{n_pos[0.0015]} of 6; "
        f"at 1 tick the 10 best days carry {_p(min(t10), 0)}-{_p(max(t10), 0)} of P&L in the winning "
        f"splits.**"
    )
    b = ast["base"]
    share_vr = ast["delta_from_vr"] / ast["delta_pnl"] if ast["delta_pnl"] < 0 else float("nan")
    tracks = share_vr >= 0.5
    c3 = (
        f"**Conclusion: P&L {'tracks VR' if tracks else 'sits below VR rather than tracking it'}. "
        f"VR explains {_p(b['r2'], 0)} of "
        f"coin-year P&L (t {_f(b['t_slope'], 1)}, Spearman {ast['spearman']:+.2f}); the drift of VR "
        f"toward 1 accounts for only {_p(share_vr, 0)} of the early-to-late P&L fall, so the decay is "
        f"mostly {'regime (VR rising)' if tracks else 'capture (less P&L at the same VR)'}, "
        f"with borderline significance (year term t {_f(ast['with_year']['year_t'], 1)}).**"
    )
    ex = [by[(s, 0.0)]["excess_over_cash"] for s in base_split]
    beats = [s for s in base_split if by[(s, 0.0)]["excess_over_cash"] > 0]
    c4 = (
        f"**Conclusion: T09 beats 4% cash in {len(beats)} of {len(base_split)} splits "
        f"({', '.join(beats) or 'none'}); the best excess is {max(ex) * 100:+.1f} pp and the worst "
        f"{min(ex) * 100:+.1f} pp.**"
    )
    return [c1, c2, c3, c4]


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--prereg", default="research/g1_preregistration.json")
    ap.add_argument("--g0", default=G0_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--raw", default=RAW_PATH, help="where the raw numbers are cached (json)")
    ap.add_argument("--reuse-raw", action="store_true", help="re-render from --raw, run nothing")
    ap.add_argument("--resume", action="store_true", help="skip sections already present in --raw")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--processes", type=int, default=1)
    args = ap.parse_args(argv)

    raw_path = Path(args.raw)
    raw: dict[str, Any] = {}
    if (args.reuse_raw or args.resume) and raw_path.exists():
        raw = json.loads(raw_path.read_text())

    def save() -> None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, indent=1, default=float))

    todo = [k for k in ("audit", "sweep", "attribution") if k not in raw]
    if todo and args.reuse_raw:
        ap.error(f"--reuse-raw: {raw_path} lacks {todo}")
    if todo:
        t_start = time.time()
        prereg = json.loads(Path(args.prereg).read_text())
        symbols = tuple(args.symbols)
        trials = g1.build_trials(prereg)
        trial = trials[8]
        assert (trial.min_step, trial.spacing_sigmas) == T09, trial.label
        splits = g1.build_splits(prereg, smoke=False)
        need = tuple(dict.fromkeys(symbols + VR_COINS))
        horizon = prereg["fixed_grid"]["sigma_horizon_minutes"]
        bars = {s: g1.load_klines(s, Path(args.data_dir)) for s in need}
        feats = precompute_features(bars, horizon)
        shared = dict(bars=bars, features=feats, filters={s: g1.filters_for(s, bars[s]) for s in bars})
        text = Path(args.g0).read_text()
        vr, vr_rob = parse_yearly_vr(text), parse_yearly_vr(text, robust=True)
        raw["symbols"] = list(symbols)

        if "audit" not in raw:
            print("1/3 exposure audit", file=sys.stderr)
            split = next(s for s in splits if s.name == AUDIT_SPLIT)
            raw["audit"] = run_audit(prereg, trial, split, symbols, shared)
            save()
        if "sweep" not in raw:
            print("2/3 fill-depth sweep", file=sys.stderr)
            raw["sweep"] = run_sweep(prereg, trial, splits, symbols, shared, args.processes)
            save()
        if "attribution" not in raw:
            print("3/3 VR attribution", file=sys.stderr)
            raw["attribution"] = run_attribution(prereg, trial, shared, vr, vr_rob, args.processes)
            raw["attribution_stats"] = attribution_stats(raw["attribution"])
            raw["runtime"] = raw.get("runtime", 0.0) + time.time() - t_start
            save()

    raw["attribution_stats"] = attribution_stats(raw["attribution"])
    zero = next(r for r in raw["sweep"] if r["split"] == AUDIT_SPLIT and r["min_pierce"] == 0.0)
    raw["audit"]["traced_equals_plain"] = bool(
        abs(raw["audit"]["final_return"] - zero["total_return"]) < 1e-12
    )
    raw["conclusions"] = conclusions(raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(raw))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
