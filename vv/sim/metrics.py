"""Performance metrics from a daily equity curve, baselines, and a Sharpe bootstrap CI.

Conventions: daily simple returns of equity; 365-day annualisation; risk-free rate 0.
A degenerate series (zero variance) has Sharpe 0.0 rather than NaN, so that a trial that
never trades sorts as "flat" instead of poisoning a median.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

ANN = 365
CASH_YIELD = 0.04  # idle-USDT benchmark, per year (compounded, 365-day years)
_BAR = pd.Timedelta(minutes=1)


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple returns between consecutive equity points."""
    return equity.astype(float).pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    """Compound annual growth over the span of the index, in 365-day years."""
    days = (equity.index[-1] - equity.index[0]) / pd.Timedelta(days=1)
    if days <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return -1.0 if equity.iloc[-1] <= 0 else 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (ANN / days) - 1.0)


def ann_vol(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * math.sqrt(ANN))


def sharpe(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * math.sqrt(ANN))


def downside_deviation(returns: pd.Series | np.ndarray) -> float:
    """sqrt(mean(min(r, 0)^2)) over *all* observations (target 0)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))


def sortino(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    dd = downside_deviation(r)
    mean = float(r.mean())
    if dd == 0.0:
        return math.inf if mean > 0 else 0.0
    return float(mean / dd * math.sqrt(ANN))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall as a positive fraction of the peak."""
    e = equity.astype(float).to_numpy()
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.max(1.0 - e / peak))


def longest_underwater_days(equity: pd.Series) -> float:
    """Longest stretch (days) from a peak until equity is back at or above it.

    An episode still open at the end of the series counts up to the last timestamp.
    """
    e = equity.astype(float).to_numpy()
    if e.size == 0:
        return 0.0
    t = equity.index
    best = pd.Timedelta(0)
    peak_i = 0
    peak = e[0]
    for i in range(1, e.size):
        if e[i] >= peak:
            if e[i - 1] < peak:
                best = max(best, t[i] - t[peak_i])
            peak, peak_i = e[i], i
    if e[-1] < peak:
        best = max(best, t[-1] - t[peak_i])
    return float(best / pd.Timedelta(days=1))


def net_capture_over_fees(gross_pnl: float, fees: float) -> float:
    """Gross realised grid PnL (before fees) per unit of fees paid; 0 if nothing was traded."""
    if fees <= 0:
        return 0.0
    return float(gross_pnl / fees)


def cash_yield_return(index: pd.DatetimeIndex, rate: float = CASH_YIELD) -> float:
    """Return of idle cash earning ``rate`` per year (compound) over the span of ``index``."""
    days = (index[-1] - index[0]) / pd.Timedelta(days=1)
    return float((1.0 + rate) ** (max(days, 0.0) / ANN) - 1.0)


def cash_yield_curve(
    times: pd.DatetimeIndex, start_cash: float, rate: float = CASH_YIELD
) -> pd.Series:
    """Equity of ``start_cash`` left in a ``rate``-per-year USDT yield, marked at ``times``."""
    years = (times - times[0]) / pd.Timedelta(days=ANN)
    return pd.Series(start_cash * (1.0 + rate) ** np.asarray(years, dtype=float), index=times,
                     name="cash_yield")


def daily_pnl(equity: pd.Series) -> pd.Series:
    """P&L between consecutive equity points (USDT)."""
    return equity.astype(float).diff().dropna()


def top_n_day_pnl_share(pnl: pd.Series | np.ndarray, n: int = 10) -> float:
    """Sum of the ``n`` best days' P&L over total P&L (can exceed 1 when other days lose).

    NaN when there is no P&L series or the total is not positive (the share of a flat or
    losing book is not meaningful).
    """
    p = np.asarray(pnl, dtype=float)
    if p.size == 0:
        return float("nan")
    total = float(p.sum())
    if not total > 0.0:
        return float("nan")
    top = np.sort(p)[::-1][: max(int(n), 0)]
    return float(top.sum() / total)


def exposure_audit(held: pd.Series, resting: pd.Series, equity: pd.Series) -> dict[str, float]:
    """Time-weighted exposure from a fine-grained trace (same index for all three series).

    ``held``: holdings at mark (USDT); ``resting``: notional of resting buys (USDT);
    ``equity``: total equity (USDT). Each sample is held until the next one, so the last
    sample carries no weight. Fractions are of *equity* (0..1, not percent).

    * ``holdings_exposure``: mean of ``held / equity``;
    * ``total_exposure``: mean of ``(held + resting) / equity`` (what the exposure cap bounds);
    * ``avg_held_usdt`` / ``avg_deployed_usdt``: the same in USDT (deployed = held + resting);
    * ``pnl``: last minus first equity;
    * ``return_on_avg_held`` / ``return_on_avg_deployed``: ``pnl`` over those averages
      (NaN when the average is zero).
    """
    nan = float("nan")
    if len(equity) < 2:
        return dict.fromkeys(
            ("holdings_exposure", "total_exposure", "avg_held_usdt", "avg_deployed_usdt", "pnl",
             "return_on_avg_held", "return_on_avg_deployed"), nan,
        )
    t = equity.index.asi8.astype(float)
    w = np.diff(t)
    w_sum = float(w.sum())
    e = equity.to_numpy(dtype=float)[:-1]
    h = held.to_numpy(dtype=float)[:-1]
    r = resting.to_numpy(dtype=float)[:-1]
    safe = np.where(e > 0, e, np.nan)
    avg_h = float((h * w).sum() / w_sum)
    avg_d = float(((h + r) * w).sum() / w_sum)
    pnl = float(equity.iloc[-1] - equity.iloc[0])
    return {
        "holdings_exposure": float(np.nansum(h / safe * w) / w_sum),
        "total_exposure": float(np.nansum((h + r) / safe * w) / w_sum),
        "avg_held_usdt": avg_h,
        "avg_deployed_usdt": avg_d,
        "pnl": pnl,
        "return_on_avg_held": pnl / avg_h if avg_h > 0 else nan,
        "return_on_avg_deployed": pnl / avg_d if avg_d > 0 else nan,
    }


def deployed_from_decisions(decisions: pd.DataFrame) -> dict[str, float]:
    """Capital tied up (holdings + resting buys, after each decision) from ``ReplayResult.decisions``.

    ``avg_total_exposure``: mean of ``exposure_total_post`` (fraction of equity);
    ``avg_deployed_usdt``: mean of ``exposure_total_post * equity``. The older ``avg_exposure``
    counts holdings only (pre-decision), which is a different and much smaller number.
    """
    if len(decisions) == 0:
        return {"avg_total_exposure": float("nan"), "avg_deployed_usdt": float("nan")}
    post = decisions["exposure_total_post"].to_numpy(dtype=float)
    eq = decisions["equity"].to_numpy(dtype=float)
    return {"avg_total_exposure": float(post.mean()), "avg_deployed_usdt": float((post * eq).mean())}


def summarize(
    equity: pd.Series,
    *,
    fees_paid: float = 0.0,
    gross_pnl: float = 0.0,
    round_trips: int = 0,
    avg_exposure: float = float("nan"),
    avg_total_exposure: float = float("nan"),
    avg_deployed_usdt: float = float("nan"),
) -> dict[str, Any]:
    r = daily_returns(equity)
    cash_ret = cash_yield_return(equity.index)
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(equity),
        "longest_underwater_days": longest_underwater_days(equity),
        "round_trips": int(round_trips),
        "fees_paid": float(fees_paid),
        "gross_pnl": float(gross_pnl),
        "net_capture_over_fees": net_capture_over_fees(gross_pnl, fees_paid),
        "avg_exposure": float(avg_exposure),
        "cash_yield_return": cash_ret,
        "excess_over_cash": total_return(equity) - cash_ret,
        "avg_total_exposure": float(avg_total_exposure),
        "avg_deployed_usdt": float(avg_deployed_usdt),
        "return_on_avg_deployed": (
            float(equity.iloc[-1] - equity.iloc[0]) / avg_deployed_usdt
            if avg_deployed_usdt > 0 else float("nan")
        ),
    }


def baseline_curves(
    bars: Mapping[str, pd.DataFrame],
    times: pd.DatetimeIndex,
    start_cash: float,
    avg_exposure: float,
) -> dict[str, pd.Series]:
    """Baseline equity curves marked at ``times`` (same marks as the replay: the close of the
    bar labelled ``t - 1min``).

    * ``cash``: flat.
    * ``equal_weight_bh``: ``start_cash`` split equally over the symbols already listed at
      ``times[0]``, bought at the first mark and held (no rebalancing; later listings ignored).
    * ``bh_scaled_to_exposure``: ``avg_exposure`` of ``start_cash`` in that basket, the rest cash.
    """
    marks: dict[str, np.ndarray] = {}
    for s, df in bars.items():
        close = df["close"].to_numpy(dtype=float)
        idx = df.index.as_unit("ns").asi8
        want = (times - _BAR).as_unit("ns").asi8
        pos = np.searchsorted(idx, want, side="right") - 1
        m_ = np.where(pos >= 0, close[np.where(pos >= 0, pos, 0)], np.nan)
        if not np.isnan(m_[0]):
            marks[s] = m_
    cash = pd.Series(float(start_cash), index=times, name="cash")
    if not marks:
        return {"cash": cash, "equal_weight_bh": cash.copy(), "bh_scaled_to_exposure": cash.copy()}
    w = start_cash / len(marks)
    basket = sum(w * (v / v[0]) for v in marks.values())
    bh = pd.Series(basket, index=times, name="equal_weight_bh")
    scaled = start_cash * (1.0 - avg_exposure) + avg_exposure * bh
    return {
        "cash": cash,
        "equal_weight_bh": bh,
        "bh_scaled_to_exposure": pd.Series(
            scaled.to_numpy(), index=times, name="bh_scaled_to_exposure"
        ),
    }


def stationary_bootstrap_indices(
    n: int, mean_block: float, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: (reps, n) indices, geometric blocks, wrap-around."""
    p = 1.0 / max(mean_block, 1.0)
    restart = rng.random((reps, n)) < p
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(reps, n))
    col = np.arange(n)
    last = np.maximum.accumulate(np.where(restart, col, 0), axis=1)  # column of the block start
    base = np.take_along_axis(starts, last, axis=1)
    return (base + (col - last)) % n


def sharpe_ci(
    daily_returns: pd.Series | np.ndarray,
    mean_block: float = 10,
    reps: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile ``1 - alpha`` CI of the annualised Sharpe under a stationary block bootstrap."""
    r = np.asarray(daily_returns, dtype=float)
    if r.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(r.size, mean_block, reps, rng)
    s = r[idx]
    sd = s.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = np.where(sd > 0, s.mean(axis=1) / sd * math.sqrt(ANN), 0.0)
    lo, hi = np.quantile(sh, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
