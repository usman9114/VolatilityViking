"""Replay engine: the production ``decide()`` against ``FakeExchange`` on historical 1m bars.

One shared USDT wallet across all symbols. Loop per decision time ``t``::

    ex.advance_to(t) -> Snapshot -> decide() -> persist halt / peak -> cancels, then places

Conventions
-----------
* Features are computed once per symbol over the **full** history (the warm-up needs data
  before ``start``) and looked up per decision: the row used at ``t`` is the latest one whose
  bar has fully closed (bar label ``< t``; labels are bar *open* times). Nothing is recomputed
  per step.
* The exchange only gets bars from ``start - 60min`` to ``end`` (matching earlier bars would
  only cost time). Bars at or after ``end`` are never visible.
* ``OrderRejected`` is counted by reason and never raised (id reuse shows up as
  ``duplicate_client_id``). Any other exception propagates.
* Daily equity is sampled at the first decision of each UTC day (before that decision's
  intents) plus a final point at ``end``. All series are marked at the exchange's last closed
  bar, i.e. the same marks ``decide`` uses.
* ``exposure`` is invested value (holdings at mark) over equity; ``exposure_total_post`` also
  counts resting buys, measured after the decision's intents were applied: it is the
  quantity the ``total_exposure_cap`` is meant to bound.
* ``gross_pnl`` is realised PnL before fees (sells vs. fee-free volume-weighted cost).
  ``round_trips`` counts sells that take a position back to zero.
* The strategy's ``GridParams.maker_fee`` is a strategy input and is *not* tied to the
  exchange ``fees`` (stress runs change only the exchange side).

Performance: the whole run is driven synchronously (the exchange coroutines never suspend),
no event loop is created.
"""

from __future__ import annotations

import multiprocessing
import time
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from vv.core.decide import decide
from vv.core.features import compute_features
from vv.core.params import GridParams, RiskParams
from vv.core.types import (
    CancelOrder,
    HaltState,
    PlaceOrder,
    Snapshot,
    SymbolFeatures,
    SymbolFilters,
)
from vv.sim.fake_exchange import FakeExchange, OrderRejected

_DAY_NS = 86_400 * 10**9
_PREROLL = pd.Timedelta(minutes=60)
_QTY_EPS = 1e-12


@dataclass(frozen=True)
class Fees:
    maker: float = 0.00075
    taker: float = 0.001
    taker_slippage_bps: float = 10.0


@dataclass
class ReplayResult:
    start: pd.Timestamp
    end: pd.Timestamp
    decision_minutes: int
    start_cash: float
    final_equity: float
    daily_equity: pd.Series  # UTC timestamps -> equity (USDT)
    decisions: pd.DataFrame  # per decision: equity, exposure, exposure_total_post, n_positions, halt_active
    n_decisions: int = 0
    n_fills: int = 0
    n_taker_fills: int = 0
    n_places: int = 0
    n_cancels: int = 0
    n_expired: int = 0
    fees_paid: float = 0.0
    gross_pnl: float = 0.0
    gross_pnl_by_symbol: dict[str, float] = field(default_factory=dict)
    round_trips: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    halt_events: list[dict[str, Any]] = field(default_factory=list)
    time_in_market: float = 0.0  # fraction of decisions holding any position
    avg_exposure: float = 0.0  # mean of ``exposure`` over decisions
    max_exposure_total_post: float = 0.0
    fills: list[dict[str, Any]] = field(default_factory=list)  # empty if keep_fills=False
    trace: list[tuple[datetime, dict[str, datetime]]] | None = None
    symbols: tuple[str, ...] = ()
    elapsed: float = 0.0  # wall seconds spent in run_replay


def _ts(x: Any) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def precompute_features(
    bars: Mapping[str, pd.DataFrame], horizon_minutes: int
) -> dict[str, pd.DataFrame]:
    """``compute_features`` per symbol over the full history (share across runs and trials)."""
    return {s: compute_features(df, horizon_minutes) for s, df in bars.items()}


def _sync(coro):
    """Run a coroutine that is known never to suspend (FakeExchange) without an event loop."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise RuntimeError("exchange coroutine suspended; replay expects a synchronous adapter")


def run_replay(
    bars: Mapping[str, pd.DataFrame],
    filters: Mapping[str, SymbolFilters],
    grid: GridParams,
    risk: RiskParams,
    start: Any,
    end: Any,
    decision_minutes: int = 15,
    start_cash: float = 10_000.0,
    fees: Fees | None = None,
    fill_haircut: float = 1.0,
    seed: int = 0,
    *,
    features: Mapping[str, pd.DataFrame] | None = None,
    record_trace: bool = False,
    keep_fills: bool = True,
) -> ReplayResult:
    """Replay ``[start, end)`` at ``decision_minutes`` steps; see the module docstring.

    ``features`` may carry ``precompute_features`` output for the full history (it must have
    been computed with ``grid.sigma_horizon_minutes``); otherwise it is computed here.
    """
    t_wall = time.perf_counter()
    fees = fees or Fees()
    start, end = _ts(start), _ts(end)
    if end <= start:
        raise ValueError("end must be after start")
    if decision_minutes < 1:
        raise ValueError("decision_minutes must be >= 1")

    # Exchange bars: only the replay window (plus a short pre-roll), only symbols with data.
    window = {
        s: df.loc[start - _PREROLL : end - pd.Timedelta(minutes=1)]
        for s, df in bars.items()
        if s in filters
    }
    window = {s: df for s, df in window.items() if len(df)}
    if not window:
        raise ValueError("no bars inside the replay window")
    symbols = tuple(sorted(window))
    if features is None:
        features = {s: compute_features(bars[s], grid.sigma_horizon_minutes) for s in symbols}

    ex = FakeExchange(
        window,
        {s: filters[s] for s in symbols},
        start_cash,
        maker_fee=fees.maker,
        taker_fee=fees.taker,
        taker_slippage_bps=fees.taker_slippage_bps,
        fill_haircut=fill_haircut,
        seed=seed,
    )

    times = pd.date_range(start, end, freq=f"{decision_minutes}min", inclusive="left")
    times_py = list(times.to_pydatetime())
    times_ns = times.as_unit("ns").asi8

    # Feature lookup, vectorised: latest row with label < t (bar closed at or before t).
    feat_cols: dict[str, dict[str, np.ndarray]] = {}
    feat_label_ns: dict[str, np.ndarray] = {}
    feat_pos: dict[str, np.ndarray] = {}
    for s in symbols:
        fdf = features[s]
        idx_ns = fdf.index.as_unit("ns").asi8
        pos = np.searchsorted(idx_ns, times_ns, side="left") - 1
        ok = np.where(pos >= 0, pos, 0)
        feat_pos[s] = pos
        feat_label_ns[s] = idx_ns[ok]
        feat_cols[s] = {
            "close": fdf["close"].to_numpy(dtype=float)[ok],
            "sigma_h": fdf["sigma_h"].to_numpy(dtype=float)[ok],
            "ema_4h": fdf["ema_4h"].to_numpy(dtype=float)[ok],
            "regime_ok": fdf["regime_ok"].to_numpy(dtype=bool)[ok],
            "warm": fdf["warm"].to_numpy(dtype=bool)[ok],
        }

    halt = HaltState()
    peak = float(start_cash)
    rejections: Counter[str] = Counter()
    halt_events: list[dict[str, Any]] = []
    trace: list[tuple[datetime, dict[str, datetime]]] | None = [] if record_trace else None

    # Fill bookkeeping (gross PnL, round trips).
    n_seen = 0
    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    gross_by_sym: dict[str, float] = {}
    fees_paid = 0.0
    round_trips = 0
    n_taker = 0

    def book_new_fills() -> None:
        nonlocal n_seen, fees_paid, round_trips, n_taker
        fills = ex.fills
        for f in fills[n_seen:]:
            s, q, px = f["symbol"], f["qty"], f["price"]
            fees_paid += f["fee"]
            if f["liquidity"] == "taker":
                n_taker += 1
            if f["side"] == "buy":
                qty[s] = qty.get(s, 0.0) + q
                cost[s] = cost.get(s, 0.0) + px * q
            else:
                held = qty.get(s, 0.0)
                avg = cost.get(s, 0.0) / held if held > 0 else px
                gross_by_sym[s] = gross_by_sym.get(s, 0.0) + (px - avg) * q
                qty[s] = held - q
                cost[s] = cost.get(s, 0.0) - avg * q
                if qty[s] <= _QTY_EPS:
                    qty[s], cost[s] = 0.0, 0.0
                    round_trips += 1
        n_seen = len(fills)

    d_ts: list[datetime] = []
    d_equity: list[float] = []
    d_expo: list[float] = []
    d_expo_post: list[float] = []
    d_npos: list[int] = []
    d_halt: list[bool] = []
    daily_ts: list[datetime] = []
    daily_eq: list[float] = []
    last_day = None
    n_places = n_cancels = 0

    for i, t in enumerate(times_py):
        ex.advance_to(t)
        book_new_fills()

        markets = _sync(ex.fetch_tickers(symbols))
        pf = ex.portfolio_state(peak, halt)
        marks = ex.marks()
        equity = pf.equity(marks)

        feats: dict[str, SymbolFeatures] = {}
        used: dict[str, datetime] = {}
        for s in markets:
            if feat_pos[s][i] < 0:
                continue
            c = feat_cols[s]
            feats[s] = SymbolFeatures(
                close=float(c["close"][i]),
                sigma_h=float(c["sigma_h"][i]),
                ema_4h=float(c["ema_4h"][i]),
                regime_ok=bool(c["regime_ok"][i]),
                warm=bool(c["warm"][i]),
            )
            if trace is not None:
                used[s] = pd.Timestamp(int(feat_label_ns[s][i]), tz="UTC").to_pydatetime()
        if trace is not None:
            trace.append((t, used))

        snapshot = Snapshot(ts=t, markets=markets, bars={}, portfolio=pf, stale=(), features=feats)
        decision = decide(snapshot, grid, risk)

        # Halt events (transitions only), then persist the halt and the equity peak.
        if decision.halt.active and not halt.active:
            halt_events.append(
                {"ts": t, "kind": "portfolio_halt", "reason": decision.halt.reason, "symbols": ()}
            )
        new_blocked = sorted(decision.halt.blocked_symbols - halt.blocked_symbols)
        if new_blocked:
            halt_events.append(
                {"ts": t, "kind": "symbol_halt", "reason": decision.halt.reason,
                 "symbols": tuple(new_blocked)}
            )
        halt = decision.halt
        peak = max(peak, equity)

        # Cancels first, then places (stable within each group).
        ordered = [x for x in decision.intents if isinstance(x, CancelOrder)] + [
            x for x in decision.intents if isinstance(x, PlaceOrder)
        ]
        for intent in ordered:
            try:
                if isinstance(intent, CancelOrder):
                    _sync(ex.cancel_order(intent))
                    n_cancels += 1
                else:
                    _sync(ex.create_order(intent))
                    n_places += 1
            except OrderRejected as e:
                rejections[e.reason] += 1

        pf_post = ex.portfolio_state(peak, halt)
        held_value = sum(p.qty * marks[p.symbol] for p in pf.positions.values() if p.qty)
        if equity > 0:
            expo, expo_post = held_value / equity, pf_post.exposure(marks) / equity
        else:
            expo = expo_post = 0.0
        d_ts.append(t)
        d_equity.append(equity)
        d_expo.append(expo)
        d_expo_post.append(expo_post)
        d_npos.append(sum(1 for p in pf.positions.values() if p.qty > 0))
        d_halt.append(halt.active)

        day = int(times_ns[i]) // _DAY_NS
        if day != last_day:
            last_day = day
            daily_ts.append(t)
            daily_eq.append(equity)

    ex.advance_to(end.to_pydatetime())
    book_new_fills()
    final_equity = ex.equity()
    daily_ts.append(end.to_pydatetime())
    daily_eq.append(final_equity)

    decisions = pd.DataFrame(
        {
            "equity": d_equity,
            "exposure": d_expo,
            "exposure_total_post": d_expo_post,
            "n_positions": d_npos,
            "halt_active": d_halt,
        },
        index=pd.DatetimeIndex(d_ts, name="ts"),
    )
    n = len(decisions)
    return ReplayResult(
        start=start,
        end=end,
        decision_minutes=decision_minutes,
        start_cash=float(start_cash),
        final_equity=float(final_equity),
        daily_equity=pd.Series(daily_eq, index=pd.DatetimeIndex(daily_ts, name="ts"), name="equity"),
        decisions=decisions,
        n_decisions=n,
        n_fills=len(ex.fills),
        n_taker_fills=n_taker,
        n_places=n_places,
        n_cancels=n_cancels,
        n_expired=len(ex.expired),
        fees_paid=float(fees_paid),
        gross_pnl=float(sum(gross_by_sym.values())),
        gross_pnl_by_symbol=dict(gross_by_sym),
        round_trips=round_trips,
        rejections=dict(rejections),
        halt_events=halt_events,
        time_in_market=float((decisions["n_positions"] > 0).mean()) if n else 0.0,
        avg_exposure=float(decisions["exposure"].mean()) if n else 0.0,
        max_exposure_total_post=float(decisions["exposure_total_post"].max()) if n else 0.0,
        fills=[dict(f) for f in ex.fills] if keep_fills else [],
        trace=trace,
        symbols=symbols,
        elapsed=time.perf_counter() - t_wall,
    )


# --------------------------------------------------------------------------- parallel
@dataclass(frozen=True)
class ReplayJob:
    """One replay. ``kwargs`` are ``run_replay`` keyword arguments (``grid``, ``risk``, ``start``,
    ``end``, ...). ``bars`` / ``features`` / ``filters`` are taken from the shared store set up by
    ``run_many`` unless given; ``symbols`` selects a subset of the shared store."""

    key: Hashable
    kwargs: dict[str, Any]
    symbols: tuple[str, ...] | None = None


_SHARED: dict[str, Any] = {}


def _run_job(job: ReplayJob) -> tuple[Hashable, ReplayResult]:
    kw = dict(job.kwargs)
    if "bars" not in kw:
        syms = job.symbols if job.symbols is not None else tuple(_SHARED["bars"])
        kw["bars"] = {s: _SHARED["bars"][s] for s in syms}
        if _SHARED.get("features") is not None and "features" not in kw:
            kw["features"] = {s: _SHARED["features"][s] for s in syms}
    kw.setdefault("filters", _SHARED.get("filters"))
    kw.setdefault("keep_fills", False)
    return job.key, run_replay(**kw)


def run_many(
    jobs: Sequence[ReplayJob],
    processes: int = 1,
    *,
    bars: Mapping[str, pd.DataFrame] | None = None,
    features: Mapping[str, pd.DataFrame] | None = None,
    filters: Mapping[str, SymbolFilters] | None = None,
) -> dict[Hashable, ReplayResult]:
    """Run jobs, optionally in ``processes`` worker processes; results keyed by ``job.key``.

    The big inputs (``bars``, ``features``, ``filters``) are placed in a module-level store
    that fork()ed workers inherit copy-on-write, so they are never pickled per job. Results do
    not depend on scheduling order or process count. Jobs are dispatched in the given order,
    so put the long ones first.
    """
    keys = [j.key for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("job keys must be unique")
    _SHARED.clear()
    _SHARED.update(bars=bars, features=features, filters=filters)
    try:
        if processes <= 1 or len(jobs) <= 1:
            return dict(_run_job(j) for j in jobs)
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(min(processes, len(jobs))) as pool:
            return dict(pool.imap_unordered(_run_job, jobs, chunksize=1))
    finally:
        _SHARED.clear()


__all__ = [
    "Fees",
    "ReplayJob",
    "ReplayResult",
    "precompute_features",
    "run_many",
    "run_replay",
]
