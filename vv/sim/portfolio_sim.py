"""Target-weight portfolio simulator: spot (long/flat) and USD-M perpetuals (long/short).

Time convention (all on a regular hourly grid of *instants*):
  * perp bars are labelled by OPEN time (the Binance 1h kline convention): the bar opened at
    T0 has its open at instant T0 and its close/high/low at instant T0 + 1h.
  * spot bars are labelled by CLOSE time (``data/futures/*_spot_1h.parquet``): the value at
    label T is the price at instant T.
A daily weight row dated t is decided at the close of day t, i.e. instant t + 1 day 00:00, and
is executed at the first instant strictly after that (t + 1 day + 1h): perp at the open of the
bar opened then, spot at the spot close with that label.

Accounting is "cash plus signed units": ``equity = cash + sum(qty * mark)``. For perps this is
identical to margin accounting (cash collateral + unrealised PnL): selling q units at P credits
cash by q*P and the open short carries -q*mark. ``funding_paid`` is the net amount paid
(positive = paid, negative = received); ``turnover`` is annualised traded notional divided by
equity at the time of each trade.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


@dataclass
class SimResult:
    equity_hourly: pd.Series  # indexed by instant
    equity_daily: pd.Series  # indexed by UTC date d, value at the close of d (instant d+1 00:00)
    trades: pd.DataFrame
    fees_paid: float
    funding_paid: float
    liquidations: list[dict]
    turnover: float
    max_gross: float
    exposure_daily: pd.DataFrame = field(default_factory=pd.DataFrame)  # net weight per coin
    coverage_warnings: int = 0


def _as_instants(df: pd.DataFrame, grid: pd.DatetimeIndex, venue: str) -> dict[str, np.ndarray]:
    """Align one coin's bars onto the instant grid."""
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    d = df.set_axis(idx)
    d = d[~d.index.duplicated(keep="last")]
    out: dict[str, np.ndarray] = {}
    if venue == "spot":
        px = d["close"].reindex(grid).to_numpy(float)
        out["exec"] = px
        out["mark"] = px
    else:
        out["exec"] = d["open"].reindex(grid).to_numpy(float)
        for src, key in (("close", "mark"), ("high", "high"), ("low", "low")):
            out[key] = d[src].reindex(grid - HOUR).to_numpy(float)
    return out


def simulate(
    weights_daily: pd.DataFrame,
    venue: Literal["spot", "perp"],
    prices_1h: dict[str, pd.DataFrame],
    funding: dict[str, pd.Series],
    *,
    fee: float,
    slippage_bps: float,
    maintenance_margin: float = 0.005,
    start_cash: float = 10_000,
    band: Callable[[pd.Series, pd.Series], pd.Series] | None = None,
) -> SimResult:
    if venue not in ("spot", "perp"):
        raise ValueError(f"venue must be 'spot' or 'perp', got {venue!r}")
    w_all = weights_daily.copy()
    if w_all.index.tz is not None:
        w_all.index = w_all.index.tz_localize(None)
    w_all = w_all.fillna(0.0)
    syms = list(w_all.columns)
    if venue == "spot":
        if (w_all.to_numpy() < -1e-12).any():
            raise ValueError("spot venue: negative weights (no shorting)")
        if (w_all.sum(axis=1).to_numpy() > 1.0 + 1e-9).any():
            raise ValueError("spot venue: gross weight above 1 (no borrowing)")
    slip = slippage_bps / 1e4
    n = len(syms)

    # ---- instant grid
    starts = [w_all.index[0] + DAY]
    ends = []
    for s in syms:
        df = prices_1h[s]
        ix = df.index.tz_localize(None) if df.index.tz is not None else df.index
        ends.append(ix.max() + (HOUR if venue == "perp" else pd.Timedelta(0)))
    grid = pd.date_range(starts[0], max(ends), freq="h")
    m = len(grid)
    arr = [_as_instants(prices_1h[s], grid, venue) for s in syms]
    exec_px = np.column_stack([a["exec"] for a in arr])
    mark_raw = np.column_stack([a["mark"] for a in arr])
    mark = pd.DataFrame(mark_raw).ffill().to_numpy()  # last known mark (NaN before first data)
    if venue == "perp":
        high = np.column_stack([a["high"] for a in arr])
        low = np.column_stack([a["low"] for a in arr])
        # funding prints: floor to the hour, mark = mark at that instant (at or before the print)
        fund = np.zeros((m, n))
        for j, s in enumerate(syms):
            fs = funding.get(s)
            if fs is None or len(fs) == 0:
                continue
            fi = fs.index.tz_localize(None) if fs.index.tz is not None else fs.index
            pos = grid.searchsorted(fi.floor("h"), side="left")
            ok = (pos >= 0) & (pos < m)
            np.add.at(fund[:, j], pos[ok], fs.to_numpy(float)[ok])
    else:
        high = low = fund = None

    # ---- decisions -> execution instants
    exec_at: dict[int, np.ndarray] = {}
    wv = w_all.to_numpy(float)
    for k, t in enumerate(w_all.index):
        e = grid.searchsorted(t + DAY + HOUR, side="left")
        if e < m:
            exec_at[e] = wv[k]

    qty = np.zeros(n)
    cash = float(start_cash)
    eq_h = np.empty(m)
    expo = np.zeros((m, n))
    fees = 0.0
    fund_paid = 0.0
    turnover_sum = 0.0
    max_gross = 0.0
    warn_count = 0
    liqs: list[dict] = []
    trades: list[tuple] = []
    dead = False

    def mtm(i: int) -> float:
        held = qty != 0
        return cash + float(np.sum(qty[held] * mark[i, held])) if held.any() else cash

    for i in range(m):
        if dead:
            eq_h[i] = 0.0
            continue
        held = qty != 0
        if venue == "perp" and held.any():
            # funding settles on the position held at the print
            f = fund[i]
            if f.any():
                pay = float(np.sum(qty[held] * mark[i, held] * f[held]))
                cash -= pay
                fund_paid += pay
            # liquidation on the adverse extreme of the bar that just closed
            adv = np.where(qty > 0, low[i], high[i])
            adv = np.where(np.isnan(adv), mark[i], adv)
            adv = np.where(held, adv, 0.0)
            eq_adv = cash + float(np.sum(qty * adv))
            gross_adv = float(np.sum(np.abs(qty) * adv))
            if eq_adv < maintenance_margin * gross_adv:
                fee_liq = fee * gross_adv
                cash = eq_adv - fee_liq
                fees += fee_liq
                for j in np.flatnonzero(held):
                    trades.append((grid[i], syms[j], -qty[j], adv[j], fee * abs(qty[j]) * adv[j], "liquidation"))
                liqs.append({"time": grid[i], "equity_before": eq_adv, "gross_notional": gross_adv,
                             "fee": fee_liq, "positions": {syms[j]: float(qty[j]) for j in np.flatnonzero(held)}})
                qty[:] = 0.0
                if cash <= 0:
                    cash = 0.0
                    dead = True
                    eq_h[i] = 0.0
                    continue
        if i in exec_at:
            target = exec_at[i]
            px = exec_px[i]
            tradable = ~np.isnan(px) & ~np.isnan(mark[i])
            valid_mark = np.where(np.isnan(mark[i]), 0.0, mark[i])
            px_eff = np.where(tradable, px, valid_mark)
            eq_x = cash + float(np.sum(qty * px_eff))
            if eq_x > 0:
                cur_w = pd.Series(qty * px_eff / eq_x, index=syms)
                tgt_w = pd.Series(np.where(tradable, target, 0.0), index=syms)
                held_w = band(cur_w, tgt_w) if band is not None else tgt_w
                held_w = np.where(tradable, held_w.to_numpy(float), cur_w.to_numpy())
                warn_count += int(np.sum(~tradable & ((np.abs(target) > 1e-12) | (qty != 0))))
                with np.errstate(invalid="ignore", divide="ignore"):
                    dq = np.where(tradable, held_w * eq_x / px_eff - qty, 0.0)
                dq[np.abs(dq * px_eff) < 1e-9] = 0.0
                sells = np.flatnonzero(dq < 0)
                buys = np.flatnonzero(dq > 0)
                for j in sells:
                    tp = px_eff[j] * (1.0 - slip)
                    fe = fee * -dq[j] * tp
                    cash += -dq[j] * tp - fe
                    fees += fe
                    turnover_sum += -dq[j] * px_eff[j] / eq_x
                    trades.append((grid[i], syms[j], dq[j], tp, fe, "rebalance"))
                if len(buys):
                    cost = sum(dq[j] * px_eff[j] * (1.0 + slip) * (1.0 + fee) for j in buys)
                    scale = 1.0
                    if venue == "spot" and cost > cash:
                        scale = max(cash, 0.0) / cost  # never spend cash we do not have
                    for j in buys:
                        q = dq[j] * scale
                        tp = px_eff[j] * (1.0 + slip)
                        fe = fee * q * tp
                        cash -= q * tp + fe
                        fees += fe
                        turnover_sum += q * px_eff[j] / eq_x
                        trades.append((grid[i], syms[j], q, tp, fe, "rebalance"))
                        dq[j] = q
                qty = qty + dq
                qty[np.abs(qty) < 1e-15] = 0.0
        eq = mtm(i)
        eq_h[i] = eq
        if eq > 0:
            held = qty != 0
            if held.any():
                notional = qty * mark[i]
                notional = np.where(held, notional, 0.0)
                expo[i] = notional / eq
                max_gross = max(max_gross, float(np.abs(expo[i]).sum()))
        else:
            dead = True

    equity_hourly = pd.Series(eq_h, index=grid, name="equity")
    # daily equity: value at instant d+1 00:00 labelled d; plus the start point labelled d0-1
    close_inst = grid[grid.hour == 0]
    eq_d = equity_hourly.loc[close_inst]
    eq_d.index = (eq_d.index - DAY).normalize()
    equity_daily = eq_d
    expo_d = pd.DataFrame(expo, index=grid, columns=syms).loc[close_inst]
    expo_d.index = equity_daily.index

    tr = pd.DataFrame(trades, columns=["time", "symbol", "qty", "price", "fee", "kind"])
    if len(tr):
        tr["notional"] = tr["qty"].abs() * tr["price"]
    years = max((grid[-1] - grid[0]).total_seconds() / (365 * 86400), 1e-9)
    return SimResult(
        equity_hourly=equity_hourly,
        equity_daily=equity_daily,
        trades=tr,
        fees_paid=float(fees),
        funding_paid=float(fund_paid),
        liquidations=liqs,
        turnover=float(turnover_sum / years),
        max_gross=float(max_gross),
        exposure_daily=expo_d,
        coverage_warnings=warn_count,
    )
