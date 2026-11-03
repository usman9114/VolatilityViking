"""Variant B: classic spot grid. Geometric levels in a fixed range; each level is an
independent buy -> sell pair (buy at level i, sell the same qty at level i + 1, then re-arm).

Pure, like ``decide``: ``decide_paired`` maps (snapshot, state) to (Decision, new state).

Layout. ``levels`` is the number of pairs n; there are n + 1 grid prices
``L_k = anchor * (1 + step) ** (k - n / 2)``, so the range is ``anchor * (1 +- step) ** (n / 2)``
(geometric) and the anchor is the middle line. Level ``i`` (0 <= i < n) buys at ``L_i`` and,
once filled ("held"), sells at ``L_{i+1}``.

Fills arrive in ``snapshot.fills_since_last`` (client ids of fills are mapped back to a
(side, level) through ``GridState.orders``, which records every order this strategy placed that
may still rest).

Choices the pre-registration leaves open (see also the WP report):
  * the anchor is the mid price the first time a symbol is warm and unblocked;
  * quote per level = symbol budget / levels (budget = ``symbol_budgets``: inverse-vol share of
    the total cap, capped at ``per_symbol_cap``); buys are finally clipped by the same global
    cap as ``decide`` (holdings at mark + resting buys);
  * a held level's sell sits at ``L_{i+1}`` but never inside the spread: it is lifted to
    ``ask + 1 tick`` when price has already run past that line;
  * a re-centre cycle only cancels our orders and rebuilds the grid; orders follow next cycle;
  * symbols that are cold (not warm) cancel their resting buys and keep their sells.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from vv.core import filters as flt
from vv.core.decide import _apply_global_cap, _marks
from vv.core.orders import anchor_bucket, is_ours, order_id, reconcile
from vv.core.params import RiskParams
from vv.core.regime import buys_allowed
from vv.core.risk import evaluate_halt, flatten_intents
from vv.core.sizing import symbol_budgets
from vv.core.types import CancelOrder, Decision, OrderIntent, PlaceOrder, Snapshot

RECENTRE_AFTER = timedelta(hours=24)
_QTY_EPS = 1e-9


class PairedGridParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    levels: int = Field(default=10, ge=2, le=100)  # number of buy->sell pairs (even)
    step: float = Field(gt=0, le=0.2)  # geometric spacing between adjacent lines


@dataclass(frozen=True)
class GridState:
    """Per-symbol grid state."""

    anchor: float
    levels: tuple[float, ...]  # n + 1 grid prices, ascending (unrounded)
    held: frozenset[int] = frozenset()  # levels whose buy filled and whose sell is pending
    out_of_range_since: datetime | None = None
    held_qty: tuple[tuple[int, float], ...] = ()  # (level, qty bought there)
    orders: tuple[tuple[str, str, int], ...] = ()  # (client_id, side, level) we placed


def build_levels(anchor: float, step: float, n: int) -> tuple[float, ...]:
    return tuple(anchor * (1.0 + step) ** (k - n / 2.0) for k in range(n + 1))


def new_grid(anchor: float, params: PairedGridParams) -> GridState:
    return GridState(anchor=anchor, levels=build_levels(anchor, params.step, params.levels))


def _apply_fills(
    gs: GridState, symbol: str, fills: tuple, position_qty: float
) -> GridState:
    """Update ``held`` / ``held_qty`` from fills (chronological), then drop what is gone."""
    by_id = {cid: (side, lvl) for cid, side, lvl in gs.orders}
    held = set(gs.held)
    qty = dict(gs.held_qty)
    seen: set[str] = set()
    for f in fills:
        if f.get("symbol") != symbol:
            continue
        cid = f.get("client_id")
        seen.add(cid)
        hit = by_id.get(cid)
        if hit is None:
            continue
        side, lvl = hit
        if side == "buy":
            held.add(lvl)
            qty[lvl] = qty.get(lvl, 0.0) + float(f["qty"])
        else:
            held.discard(lvl)
            qty.pop(lvl, None)
    if position_qty <= _QTY_EPS:  # flat: nothing can still be pending a sell
        held.clear()
        qty.clear()
    orders = tuple(o for o in gs.orders if o[0] not in seen)
    return GridState(
        gs.anchor, gs.levels, frozenset(held), gs.out_of_range_since,
        tuple(sorted((k, v) for k, v in qty.items() if k in held)), orders,
    )


def _prune_orders(gs: GridState, open_ids: set[str]) -> GridState:
    """Forget ids that no longer rest (cancelled or rejected earlier); fills were applied first."""
    kept = tuple(o for o in gs.orders if o[0] in open_ids)
    return gs if len(kept) == len(gs.orders) else _replace_orders(gs, kept)


def _replace_orders(gs: GridState, orders: tuple[tuple[str, str, int], ...]) -> GridState:
    return GridState(
        gs.anchor, gs.levels, gs.held, gs.out_of_range_since, gs.held_qty, orders
    )


def decide_paired(
    snapshot: Snapshot,
    state: Mapping[str, GridState] | None,
    params: PairedGridParams,
    risk: RiskParams,
) -> tuple[Decision, dict[str, GridState]]:
    """Intents plus the new per-symbol grid state (the caller feeds it back next call).

    Stale data: no intents (fills are still absorbed into the state). Active halt: cancel
    everything and flatten (same latched logic as production). Otherwise per warm, unblocked
    symbol: buys at free levels below the bid (regime gate), sells for held levels, reconciled
    against resting orders and clipped by the global exposure cap.
    """
    pf = snapshot.portfolio
    states: dict[str, GridState] = dict(state or {})

    for s in list(states):  # absorb fills first, whatever else happens this cycle
        pos = pf.positions.get(s)
        states[s] = _apply_fills(
            states[s], s, snapshot.fills_since_last, pos.qty if pos else 0.0
        )

    if snapshot.stale:
        return Decision((), pf.halt), states

    marks = _marks(snapshot)
    halt = evaluate_halt(pf, marks, risk, snapshot.ts)
    markets = snapshot.markets

    if halt.active:
        everything = set(markets) | set(pf.positions) | {o.symbol for o in pf.open_orders}
        intents = flatten_intents(pf, markets, everything, snapshot.ts)
        return Decision(tuple(intents), halt), states

    equity = pf.equity(marks)
    blocked = halt.blocked_symbols
    universe = sorted(s for s in markets if s not in blocked)
    warm = [s for s in universe if (f := snapshot.features.get(s)) is not None and f.warm]
    sigmas = {s: snapshot.features[s].sigma_h for s in warm}
    budgets = symbol_budgets(equity, sigmas, risk)

    minute = int(snapshot.ts.timestamp() // 60)
    desired: list[PlaceOrder] = []
    desired_level: dict[str, tuple[str, str, int]] = {}  # client id -> (symbol, side, level)
    recentred: list[CancelOrder] = []
    recentred_syms: set[str] = set()
    open_ids = {o.client_id for o in pf.open_orders}

    for s in warm:
        m, feat = markets[s], snapshot.features[s]
        flt_ = m.filters
        mid = (m.bid + m.ask) / 2.0
        pos = pf.positions.get(s)
        pos_qty = pos.qty if pos is not None else 0.0
        gs = states.get(s)
        if gs is None:
            gs = new_grid(mid, params)
        gs = _prune_orders(gs, open_ids)

        # Range tracking and re-centring (flat and out of range for 24h).
        outside = not (gs.levels[0] <= mid <= gs.levels[-1])
        since = gs.out_of_range_since
        if not outside:
            since = None
        elif since is None:
            since = snapshot.ts
        flat = pos_qty <= _QTY_EPS and not gs.held
        if outside and flat and since is not None and snapshot.ts - since >= RECENTRE_AFTER:
            mine = [o for o in pf.open_orders if o.symbol == s and is_ours(o.client_id)]
            recentred.extend(CancelOrder(s, o.client_id) for o in mine)
            recentred_syms.add(s)
            states[s] = new_grid(mid, params)
            continue
        gs = GridState(
            gs.anchor, gs.levels, gs.held, since, gs.held_qty, gs.orders
        )
        states[s] = gs

        n = params.levels
        bucket = anchor_bucket(gs.anchor, params.step)
        quote = budgets.get(s, 0.0) / n
        held_qty = dict(gs.held_qty)

        if buys_allowed(feat):
            for i in range(n):
                if i in gs.held:
                    continue
                p = flt.round_price("buy", gs.levels[i], flt_.tick_size)
                if p <= 0 or p >= m.bid:
                    continue
                q = flt.round_qty(quote / p, flt_.step_size)
                if q > 0:
                    cid = order_id(s, "buy", i, p, q, bucket, minute)
                    desired.append(PlaceOrder(s, "buy", p, q, cid))
                    desired_level[cid] = (s, "buy", i)

        foreign_sells = sum(
            o.remaining
            for o in pf.open_orders
            if o.symbol == s and o.side == "sell" and not is_ours(o.client_id)
        )
        free = max(0.0, pos_qty - foreign_sells)
        for i in sorted(gs.held):
            q = flt.round_qty(min(held_qty.get(i, 0.0), free), flt_.step_size)
            if q <= 0:
                continue
            free -= q
            p = flt.round_price("sell", gs.levels[i + 1], flt_.tick_size)
            p = max(p, flt.round_price("sell", m.ask + flt_.tick_size, flt_.tick_size))
            cid = order_id(s, "sell", i + 1, p, q, bucket, minute)
            desired.append(PlaceOrder(s, "sell", p, q, cid))
            desired_level[cid] = (s, "sell", i)

    warm_set = set(warm)
    scoped = tuple(
        o
        for o in pf.open_orders
        if (o.symbol in warm_set and o.symbol not in recentred_syms)
        or (
            o.symbol in markets
            and o.symbol not in blocked
            and o.symbol not in warm_set
            and o.side == "buy"
        )
    )
    intents = reconcile(desired, scoped, markets=markets)
    flatten = flatten_intents(pf, markets, blocked, snapshot.ts)
    intents = _apply_global_cap(intents, snapshot, marks, equity, risk, budgets)

    # Remember which of our orders will rest, so next cycle's fills map back to levels.
    cancelled = {i.client_id for i in intents if isinstance(i, CancelOrder)}
    placed: dict[str, list[tuple[str, str, int]]] = {}
    for i in intents:
        if isinstance(i, PlaceOrder) and i.client_id in desired_level:
            sym, side, lvl = desired_level[i.client_id]
            placed.setdefault(sym, []).append((i.client_id, side, lvl))
    for s in warm:
        if s in recentred_syms:
            continue
        gs = states[s]
        kept = [o for o in gs.orders if o[0] not in cancelled]
        states[s] = _replace_orders(gs, tuple(kept + placed.get(s, [])))

    out: list[OrderIntent] = list(flatten) + list(recentred) + list(intents)
    return Decision(tuple(out), halt), states


@dataclass(frozen=True)
class PairedGridStrategy:
    """Picklable ``strategy`` for ``run_replay``: ``(snapshot, state) -> (Decision, state)``."""

    params: PairedGridParams
    risk: RiskParams

    def __call__(self, snapshot: Snapshot, state):
        return decide_paired(snapshot, state, self.params, self.risk)


__all__ = [
    "GridState",
    "PairedGridParams",
    "PairedGridStrategy",
    "build_levels",
    "decide_paired",
    "new_grid",
]
