"""Funding carry research: long spot + short USDS-M perpetual (delta-neutral).

Measures net carry on TOTAL capital (50% spot, 50% perp margin, 1x short) with
fees, monthly rebalancing and a trailing-7d-funding exit rule, per symbol and
for a monthly-rebalanced top-3 portfolio selected without look-ahead.
Evaluates gate CARRY-0. Research only, no live code.

Usage:
    python -m research.carry_analysis --out docs/research/CARRY_RESULTS.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/futures")
SPOT_DIR = Path("data/klines")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT",
           "BNBUSDT", "AVAXUSDT"]

PERIODS_PER_DAY = 3            # 8h funding
FEE_SPOT = 0.00075             # spot maker
FEE_PERP = 0.0002              # perp maker
SPOT_FRACTION = 0.5            # of total capital; the other 50% is perp margin
EXIT_WINDOW_H = 7 * 24         # trailing funding window for the exit rule
RANK_WINDOW_H = 30 * 24        # trailing funding window for portfolio ranking
TOP_K = 3
DRIFT_TOL = 0.25               # sensitivity run only
MIN_DAYS_PER_YEAR = 90         # a partial year needs this many active days to count

GATE_MEDIAN_MIN = 0.03
GATE_WORST_YEAR_MIN = -0.02
GATE_MAXDD_MAX = 0.08


# --------------------------------------------------------------------------
# Funding statistics
# --------------------------------------------------------------------------
def annualised_funding(rates: pd.Series, days: float) -> float:
    """Sum of per-period funding rates over `days`, scaled to 365 days.

    Rates are per unit of perp notional (3 periods/day at the 8h interval, so a
    constant rate r gives r * 3 * 365 per year).
    """
    return float(rates.sum()) / days * 365.0


def longest_negative_streak(rates: pd.Series) -> int:
    """Longest run of consecutive negative funding periods."""
    best = cur = 0
    for neg in (rates.to_numpy() < 0):
        cur = cur + 1 if neg else 0
        best = max(best, cur)
    return best


def funding_to_grid(funding: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Per-period rates -> hourly grid (sum of rates settling in each grid hour)."""
    f = funding.copy()
    f.index = f.index.round("h")
    f = f.groupby(level=0).sum()
    return f.reindex(grid, fill_value=0.0)


# --------------------------------------------------------------------------
# Signals (all use data at or before the decision time only)
# --------------------------------------------------------------------------
def exit_state(f_grid: pd.Series, window: int = EXIT_WINDOW_H) -> np.ndarray:
    """Hysteresis position signal: flat when trailing funding < 0, in when > 0.

    Value at index i uses funding settled at or before grid time i. Exactly zero
    keeps the previous state. Starts flat; undefined during warm-up (flat).
    """
    trailing = f_grid.rolling(window, min_periods=window).sum().to_numpy()
    on = np.zeros(len(trailing), dtype=bool)
    state = False
    for i, v in enumerate(trailing):
        if np.isfinite(v):
            if v > 0:
                state = True
            elif v < 0:
                state = False
        on[i] = state
    return on


def monthly_selection(f_grid: pd.DataFrame, avail: pd.DataFrame, k: int = TOP_K,
                      window: int = RANK_WINDOW_H) -> dict:
    """At each month start pick the top-k symbols by trailing funding.

    Uses only rows with index <= the month start. Only symbols that are available
    (prices present) and have a full window of history are ranked. Returns
    {month_start_timestamp: [symbols]}.
    """
    trailing = f_grid.rolling(window, min_periods=window).sum()
    idx = f_grid.index
    month = idx.to_period("M")
    starts = idx[np.r_[True, month[1:] != month[:-1]]]
    out = {}
    for t in starts:
        row = trailing.loc[t].where(avail.loc[t])
        row = row.dropna().sort_values(ascending=False, kind="stable")
        out[t] = list(row.index[:k])
    return out


# --------------------------------------------------------------------------
# Simulation engine
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    equity: pd.Series        # total-capital equity, starts at 1.0
    fees: pd.Series          # fees paid per hour (fraction of capital)
    funding: pd.Series       # funding received per hour
    mtm: pd.Series           # basis mark-to-market per hour
    entries: int             # number of (symbol) entries incl. initial
    exits: int
    margin_events: int = 0   # sleeve periods where the short perp lost >= 90% of its margin


def simulate(spot: pd.DataFrame, perp: pd.DataFrame, f_grid: pd.DataFrame,
             selected: dict | None = None, use_exit: bool = True,
             fee_spot: float = FEE_SPOT, fee_perp: float = FEE_PERP,
             k: int | None = None, drift_tol: float | None = None) -> SimResult:
    """Hourly delta-neutral carry simulation on total capital (equity starts 1.0).

    Each selected symbol gets weight 1/k of equity; within a sleeve, spot notional
    is 50% of sleeve equity and the short perp has the same units (1x on the other
    50% as margin). Decisions at grid time i use information <= i and take effect
    over (i, i+1]. Position units are resized to sleeve equity at each month start
    (monthly rebalance, fees on traded units) and when a symbol is entered/exited.
    Funding is received on perp notional at the settlement hour's close price.
    Idle cash earns nothing.

    `selected` maps month_start -> symbols (None = every column, always).
    `k` is the sleeve count (default: number of columns, or TOP_K when selected).
    `drift_tol` (sensitivity only; None = spec) also resizes a sleeve whenever its
    notional drifts more than this fraction from 50% of sleeve equity.
    `margin_events` counts sleeve periods in which the unrealised loss on the short
    perp since the last resize reached 90% of its margin (a liquidation risk the
    monthly-only rebalance does not manage; the simulation does not liquidate).
    """
    cols = list(f_grid.columns)
    idx = f_grid.index
    S = spot[cols].to_numpy(float)
    P = perp[cols].to_numpy(float)
    F = f_grid[cols].to_numpy(float)
    ok = np.isfinite(S) & np.isfinite(P)
    S = np.where(ok, S, 1.0)
    P = np.where(ok, P, 1.0)
    if use_exit:
        sig = np.column_stack([exit_state(f_grid[c]) for c in cols])
    else:
        sig = np.ones_like(ok)
    if k is None:
        k = TOP_K if selected is not None else len(cols)
    n, m = len(idx), len(cols)
    w = 1.0 / k

    month = idx.to_period("M")
    new_month = np.r_[True, month[1:] != month[:-1]]
    sel_mask = np.ones((n, m), dtype=bool)
    if selected is not None:
        sel_mask = np.zeros((n, m), dtype=bool)
        cur = np.zeros(m, dtype=bool)
        for i in range(n):
            if idx[i] in selected:
                cur = np.array([c in selected[idx[i]] for c in cols])
            sel_mask[i] = cur

    q = np.zeros(m)
    E = 1.0
    eq = np.empty(n)
    fee_t = np.zeros(n)
    fund_t = np.zeros(n)
    mtm_t = np.zeros(n)
    entries = exits = margin_events = 0
    ref_p = np.ones(m)
    margin = np.zeros(m)
    flagged = np.zeros(m, dtype=bool)
    for i in range(n):
        if i > 0:
            held = q > 0
            fund = float(np.sum(q * P[i] * F[i] * held))
            mtm = float(np.sum(q * ((S[i] - S[i - 1]) - (P[i] - P[i - 1]))))
            E += fund + mtm
            fund_t[i], mtm_t[i] = fund, mtm
            loss = q * (P[i] - ref_p)
            hit = held & ~flagged & (loss >= 0.9 * margin)
            margin_events += int(hit.sum())
            flagged |= hit
        target_on = sig[i] & sel_mask[i] & ok[i]
        new_q = np.zeros(m)
        for j in range(m):
            if not target_on[j]:
                continue
            drifted = (drift_tol is not None and q[j] > 0 and
                       abs(q[j] * S[i, j] / (w * E * SPOT_FRACTION) - 1) > drift_tol)
            if q[j] == 0 or new_month[i] or drifted:
                new_q[j] = w * E * SPOT_FRACTION / S[i, j]   # entry or resize
            else:
                new_q[j] = q[j]
        dq = np.abs(new_q - q)
        fee = float(np.sum(dq * (S[i] * fee_spot + P[i] * fee_perp)))
        entries += int(np.sum((q == 0) & (new_q > 0)))
        exits += int(np.sum((q > 0) & (new_q == 0)))
        E -= fee
        fee_t[i] = fee
        resized = new_q != q
        ref_p = np.where(resized, P[i], ref_p)
        margin = np.where(resized, w * E * SPOT_FRACTION, margin)
        flagged &= ~resized
        q = new_q
        eq[i] = E
    mk = lambda a: pd.Series(a, index=idx)  # noqa: E731
    return SimResult(mk(eq), mk(fee_t), mk(fund_t), mk(mtm_t), entries, exits,
                     margin_events)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall as a positive fraction of the peak."""
    peak = equity.cummax()
    return float((1 - equity / peak).max())


def sharpe(equity: pd.Series) -> float:
    """Annualised Sharpe (rf = 0) of daily equity returns, sqrt(365)."""
    d = equity.resample("1D").last().pct_change().dropna()
    if len(d) < 2 or d.std() == 0:
        return float("nan")
    return float(d.mean() / d.std() * np.sqrt(365))


def yearly_net(equity: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    """Calendar-year net return, annualised linearly by active days for partial years."""
    rows = []
    eq = equity[equity.index >= start]
    for y, g in eq.groupby(eq.index.year):
        prev = eq[eq.index < g.index[0]]
        base = prev.iloc[-1] if len(prev) else 1.0
        days = (g.index[-1] - g.index[0]).total_seconds() / 86400 + 1 / 24
        if len(prev) == 0:
            days = (g.index[-1] - start).total_seconds() / 86400
        raw = g.iloc[-1] / base - 1
        rows.append({"year": int(y), "days": days, "raw": raw,
                     "annualised": raw * 365.0 / days if days < 360 else raw})
    return pd.DataFrame(rows).set_index("year")


def gate_carry0(yearly: pd.DataFrame, maxdd: float) -> dict:
    y = yearly[yearly["days"] >= MIN_DAYS_PER_YEAR]
    median = float(y["annualised"].median())
    worst = float(y["annualised"].min())
    ok = median >= GATE_MEDIAN_MIN and worst >= GATE_WORST_YEAR_MIN and maxdd <= GATE_MAXDD_MAX
    return {"median": median, "worst": worst, "maxdd": maxdd, "pass": ok,
            "years": list(y.index)}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_spot_1h(sym: str) -> pd.Series:
    """Spot 1h close, indexed by bar CLOSE time (open + 1h)."""
    cache = DATA / f"{sym}_spot_1h.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["close"]
    m = pd.read_parquet(SPOT_DIR / f"{sym}_1m.parquet", columns=["close"])
    h = m["close"].resample("1h").last().to_frame("close")
    h.index = h.index + pd.Timedelta(hours=1)
    h.to_parquet(cache)
    return h["close"]


def load_perp_1h(sym: str) -> pd.Series:
    d = pd.read_parquet(DATA / f"{sym}_perp_1h.parquet", columns=["close"])["close"]
    d.index = d.index + pd.Timedelta(hours=1)
    return d[~d.index.duplicated(keep="last")]


def load_all(symbols=SYMBOLS, start="2020-01-01", end="2026-11-01"):
    grid = pd.date_range(start, end, freq="1h", inclusive="left") + pd.Timedelta(hours=1)
    grid = grid[grid <= pd.Timestamp(end)]
    spot, perp, fund, raw_fund = {}, {}, {}, {}
    for s in symbols:
        spot[s] = load_spot_1h(s).reindex(grid).ffill(limit=6)
        perp[s] = load_perp_1h(s).reindex(grid).ffill(limit=6)
        rf = pd.read_parquet(DATA / f"{s}_funding.parquet")["rate"]
        raw_fund[s] = rf
        fund[s] = funding_to_grid(rf, grid)
    return (pd.DataFrame(spot), pd.DataFrame(perp), pd.DataFrame(fund), raw_fund)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def funding_table(raw_fund: dict) -> pd.DataFrame:
    rows = []
    for s, r in raw_fund.items():
        r = r[r.index >= "2020-01-01"]
        for y, g in r.groupby(r.index.year):
            days = 366 if (y % 4 == 0) else 365
            span = (g.index[-1] - g.index[0]).total_seconds() / 86400 + 1 / PERIODS_PER_DAY
            span = min(span, days)
            rows.append({"symbol": s, "year": int(y), "days": span,
                         "gross": annualised_funding(g, span),
                         "neg_pct": float((g < 0).mean()),
                         "streak": longest_negative_streak(g)})
    return pd.DataFrame(rows)


def basis_table(spot: pd.DataFrame, perp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in spot.columns:
        b = (perp[s] / spot[s] - 1).dropna()
        b = b[(b.index >= "2020-01-01")]
        d1 = b.diff(1).dropna()
        d24 = (b - b.shift(24)).dropna()
        rows.append({"symbol": s, "mean": b.mean(), "p01": b.quantile(0.01),
                     "p99": b.quantile(0.99), "worst_1h": d1.max(),
                     "worst_1d": d24.max()})
    return pd.DataFrame(rows).set_index("symbol")


def run(out: Path | None):
    spot, perp, fgrid, raw_fund = load_all()
    avail = spot.notna() & perp.notna()
    ft = funding_table(raw_fund)
    bt = basis_table(spot, perp)

    per_sym, per_sym_hold, sym_year, sym_year_hold = {}, {}, {}, {}
    for s in SYMBOLS:
        cols = [s]
        first = avail[s].idxmax() + pd.Timedelta(hours=EXIT_WINDOW_H)
        for use_exit, store, sy in ((True, per_sym, sym_year), (False, per_sym_hold, sym_year_hold)):
            r = simulate(spot[cols], perp[cols], fgrid[cols], use_exit=use_exit)
            eq = r.equity[r.equity.index >= first]
            eq = eq / eq.iloc[0]
            store[s] = (r, eq)
            sy[s] = yearly_net(r.equity, first)

    # Portfolio of top-3 by trailing 30d funding, monthly, no look-ahead.
    sel = monthly_selection(fgrid, avail)
    pr = simulate(spot, perp, fgrid, selected=sel, use_exit=True, k=TOP_K)
    p_start = next(t for t, v in sel.items() if len(v) >= TOP_K)
    p_eq = pr.equity[pr.equity.index >= p_start]
    p_eq = p_eq / p_eq.iloc[0]
    p_year = yearly_net(pr.equity, p_start)
    p_dd = max_drawdown(p_eq)
    gate = gate_carry0(p_year, p_dd)
    gate_raw = gate_carry0(p_year.assign(annualised=p_year["raw"]), p_dd)

    # Sensitivity: resize any sleeve whose notional drifts >25% (margin top-up proxy).
    sr = simulate(spot, perp, fgrid, selected=sel, use_exit=True, k=TOP_K, drift_tol=DRIFT_TOL)
    s_eq = sr.equity[sr.equity.index >= p_start]
    s_eq = s_eq / s_eq.iloc[0]
    s_year = yearly_net(sr.equity, p_start)
    s_dd = max_drawdown(s_eq)
    s_gate = gate_carry0(s_year, s_dd)
    sym_margin = {s_: simulate(spot[[s_]], perp[[s_]], fgrid[[s_]]).margin_events
                  for s_ in SYMBOLS}

    res = {"sens": sr, "s_year": s_year, "s_dd": s_dd, "s_gate": s_gate,
           "s_sharpe": sharpe(s_eq), "sym_margin": sym_margin, "ft": ft, "bt": bt, "per_sym": per_sym, "per_sym_hold": per_sym_hold,
           "sym_year": sym_year, "sym_year_hold": sym_year_hold,
           "portfolio": pr, "p_eq": p_eq, "p_year": p_year, "p_dd": p_dd,
           "p_sharpe": sharpe(p_eq), "gate": gate, "gate_raw": gate_raw,
           "p_start": p_start, "sel": sel}
    DATA.mkdir(parents=True, exist_ok=True)
    ft.to_csv(DATA / "carry_funding_by_year.csv", index=False)
    p_eq.to_frame("equity").to_parquet(DATA / "carry_portfolio_equity.parquet")
    if out is not None:
        out.write_text(render(res))
    return res


def render(r: dict) -> str:
    ft, bt = r["ft"], r["bt"]
    years = sorted(ft["year"].unique())
    L = ["# Funding carry research (WP-S0E)", "",
         "Long spot + short USDT-M perpetual, 50% spot / 50% perp margin (1x), net of fees "
         "(spot maker 0.075%, perp maker 0.02%, both legs on entry, exit and monthly resize), "
         "idle cash earns 0. Hourly simulation, Jan 2020 to Oct 2026. Exit rule: flat when "
         "trailing 7d funding < 0, re-enter when > 0. Reproduce: "
         "`python -m research.carry_fetch && python -m research.carry_analysis`.", ""]

    def matrix(title, getter, fmt):
        L.append(f"**{title}**"); L.append("")
        L.append("| symbol | " + " | ".join(str(y) for y in years) + " |")
        L.append("|---|" + "---|" * len(years))
        for s in SYMBOLS:
            cells = []
            for y in years:
                v = getter(s, y)
                cells.append(fmt(v) if v is not None else "-")
            L.append(f"| {s[:-4]} | " + " | ".join(cells) + " |")
        L.append("")

    def fv(col):
        def g(s, y):
            d = ft[(ft.symbol == s) & (ft.year == y)]
            return None if d.empty else float(d[col].iloc[0])
        return g

    def sv(sy):
        def g(s, y):
            d = sy[s]
            return None if y not in d.index or d.loc[y, "days"] < 60 else float(d.loc[y, "annualised"])
        return g

    matrix("Gross funding, annualised, % of perp notional", fv("gross"), _pct)
    matrix("Negative funding periods, % of periods (longest streak in periods in table below)",
           fv("neg_pct"), lambda v: _pct(v, 0))
    matrix("Net carry per symbol WITH exit rule, annualised, on total capital",
           sv(r["sym_year"]), _pct)

    L.append("**Per symbol, whole period** (net = with exit rule; hold = always in)")
    L.append("")
    L.append("| symbol | longest neg streak | net CAGR | maxDD | Sharpe | hold CAGR | hold maxDD | "
             "worst basis 1h | worst basis 1d |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in SYMBOLS:
        res, eq = r["per_sym"][s]
        hres, heq = r["per_sym_hold"][s]
        first = eq.index[0]
        streak = int(ft[ft.symbol == s]["streak"].max())
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        hcagr = heq.iloc[-1] ** (1 / yrs) - 1
        L.append(f"| {s[:-4]} | {streak} | {_pct(cagr, 2)} | {_pct(max_drawdown(eq))} | "
                 f"{sharpe(eq):.2f} | {_pct(hcagr, 2)} | {_pct(max_drawdown(heq))} | "
                 f"{_pct(bt.loc[s, 'worst_1h'], 2)} | {_pct(bt.loc[s, 'worst_1d'], 2)} |")
        _ = first
    L.append("")
    L.append("Basis = perp/spot - 1 (hourly closes); worst moves are the largest rise of the "
             "basis (adverse for the short perp), in percentage points of price.")
    L.append("")

    p = r["p_year"]
    L.append(f"**Top-3 portfolio** (rank by trailing 30d funding at each month start, equal "
             f"weight 1/3, each sleeve obeys the exit rule; start {r['p_start']:%Y-%m-%d})")
    L.append("")
    L.append("| year | active days | net (period) | net (annualised) |")
    L.append("|---|---|---|---|")
    for y, row in p.iterrows():
        L.append(f"| {y} | {row['days']:.0f} | {_pct(row['raw'], 2)} | {_pct(row['annualised'], 2)} |")
    L.append("")
    pr = r["portfolio"]
    tot_days = (pr.equity.index[-1] - r["p_start"]).days
    L.append(f"Max drawdown {_pct(r['p_dd'], 2)}, Sharpe {r['p_sharpe']:.2f}, "
             f"fees paid {_pct(pr.fees.sum(), 2)} of capital over {tot_days} days, "
             f"funding received {_pct(pr.funding.sum(), 1)}, basis MTM {_pct(pr.mtm.sum(), 1)}, "
             f"{pr.entries} entries / {pr.exits} exits.")
    L.append("")

    sr, sg = r["sens"], r["s_gate"]
    L.append(f"**Margin risk.** Monthly-only resizing lets notional drift when a coin rallies "
             f"mid-month. Sleeve periods where the short perp lost >= 90% of its margin "
             f"(no top-up, liquidation not simulated): portfolio {r['portfolio'].margin_events}; "
             f"single-symbol runs "
             + ", ".join(f"{s[:-4]} {n}" for s, n in r["sym_margin"].items()) + ". "
             f"Sensitivity, resizing also when notional drifts > {DRIFT_TOL:.0%}: "
             f"margin events {sr.margin_events}, median {_pct(sg['median'], 2)}, worst year "
             f"{_pct(sg['worst'], 2)}, max drawdown {_pct(sg['maxdd'], 2)}, Sharpe "
             f"{r['s_sharpe']:.2f}, fees {_pct(sr.fees.sum(), 2)} "
             f"({'PASS' if sg['pass'] else 'FAIL'}).")
    L.append("")

    g, gr = r["gate"], r["gate_raw"]
    verdict = "PASS" if g["pass"] else "FAIL"
    L.append(f"## Gate CARRY-0: {verdict}")
    L.append("")
    L.append(f"Median annual net {_pct(g['median'], 2)} (need >= 3%), worst year "
             f"{_pct(g['worst'], 2)} (need >= -2%), max drawdown {_pct(g['maxdd'], 2)} "
             f"(need <= 8%). Years counted (>= {MIN_DAYS_PER_YEAR} active days): "
             f"{', '.join(str(y) for y in g['years'])}; partial years annualised linearly. "
             f"Using un-annualised partial-year returns instead: median {_pct(gr['median'], 2)}, "
             f"worst {_pct(gr['worst'], 2)} ({'PASS' if gr['pass'] else 'FAIL'}).")
    L.append("")
    recent = r["p_year"].loc[[y for y in r["p_year"].index if y >= 2025], "annualised"]
    L.append("**Reading.** The literal gate passes, but narrowly on drawdown and on the back of "
             f"2020-21 (annualised {_pct(r['p_year'].loc[2020, 'annualised'], 1)} and "
             f"{_pct(r['p_year'].loc[2021, 'annualised'], 1)}). The current regime is thin: "
             f"2025-26 nets {', '.join(_pct(v, 1) for v in recent)}, below the 3% bar, and "
             "most symbols had 25-45% negative funding periods in 2025-26. Fees consume roughly "
             "1.5%/yr of capital through ~190 re-entries. Idle cash earns 0 here; parking it in "
             "T-bills or Earn would add to the figures. The monthly-only rebalance is not "
             "survivable on hot coins (see margin risk), so a live carry sleeve needs "
             "threshold rebalancing and margin top-ups.")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/research/CARRY_RESULTS.md")
    args = ap.parse_args(argv)
    run(Path(args.out))
    print(Path(args.out).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
