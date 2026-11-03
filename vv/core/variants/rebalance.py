"""Variant C: threshold rebalancing toward inverse-vol target weights. No regime gate.

Target value per coin = its ``symbol_budgets`` share (inverse-vol share of
``total_exposure_cap * equity``, capped at ``per_symbol_cap * equity``); the rest stays USDT.
When ``|value - target| > band * target`` one post-only limit is placed: a buy at
``bid - 1 tick`` or a sell at ``ask + 1 tick``, sized to close the gap and rounded down.

At most one resting order per coin (extra orders are cancelled by ``reconcile``). A resting
order is kept while it still points the right way and the target it implies has moved by no
more than ``band / 2``; otherwise it is replaced. Implied target of a resting order at limit
``p`` with remaining ``q``: ``(held qty +- q) * p``. It is stateless: everything is derived from
the snapshot.

Choices the pre-registration leaves open: "warm" (features) is still required, because the
inverse-vol weights need ``sigma_h`` (so C starts trading after the same warm-up as A and B);
cold symbols have their resting orders cancelled; positions are marked at mid; halts, stale
handling and the global cap are those of production.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from vv.core import filters as flt
from vv.core.decide import _apply_global_cap, _marks
from vv.core.orders import is_ours, order_id, reconcile
from vv.core.params import RiskParams
from vv.core.risk import evaluate_halt, flatten_intents
from vv.core.sizing import symbol_budgets
from vv.core.types import Decision, PlaceOrder, Snapshot


class RebalanceParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    band: float = Field(gt=0, lt=1)  # relative deviation from target that triggers an order


def decide_rebalance(snapshot: Snapshot, params: RebalanceParams, risk: RiskParams) -> Decision:
    """Pure map from a snapshot to at most one post-only order per coin plus the halt state."""
    pf = snapshot.portfolio
    if snapshot.stale:
        return Decision((), pf.halt)

    marks = _marks(snapshot)
    halt = evaluate_halt(pf, marks, risk, snapshot.ts)
    markets = snapshot.markets

    if halt.active:
        everything = set(markets) | set(pf.positions) | {o.symbol for o in pf.open_orders}
        return Decision(tuple(flatten_intents(pf, markets, everything, snapshot.ts)), halt)

    equity = pf.equity(marks)
    blocked = halt.blocked_symbols
    universe = sorted(s for s in markets if s not in blocked)
    warm = [s for s in universe if (f := snapshot.features.get(s)) is not None and f.warm]
    sigmas = {s: snapshot.features[s].sigma_h for s in warm}
    targets = symbol_budgets(equity, sigmas, risk)

    minute = int(snapshot.ts.timestamp() // 60)
    desired: list[PlaceOrder] = []
    for s in warm:
        m = markets[s]
        f = m.filters
        target = targets.get(s, 0.0)
        pos = pf.positions.get(s)
        qty_held = pos.qty if pos is not None else 0.0
        diff = target - qty_held * marks[s]
        if target <= 0 or abs(diff) <= params.band * target:
            continue

        side = "buy" if diff > 0 else "sell"
        if side == "buy":
            price = flt.round_price("buy", m.bid - f.tick_size, f.tick_size)
            qty = flt.round_qty(diff / price, f.step_size) if price > 0 else 0.0
        else:
            price = flt.round_price("sell", m.ask + f.tick_size, f.tick_size)
            foreign = sum(
                o.remaining
                for o in pf.open_orders
                if o.symbol == s and o.side == "sell" and not is_ours(o.client_id)
            )
            free = max(0.0, qty_held - foreign)
            qty = flt.round_qty(min(-diff / price, free), f.step_size)
        if price <= 0 or not flt.validate_notional(price, qty, f):
            continue

        keep = _keepable(pf.open_orders, s, side, qty_held, target, params.band)
        if keep is not None:
            desired.append(PlaceOrder(s, side, keep.price, keep.remaining, keep.client_id))
        else:
            cid = order_id(s, side, 0, price, qty, 0, minute)
            desired.append(PlaceOrder(s, side, price, qty, cid))

    # Every order of ours on a tradable symbol is in scope: whatever is not desired is cancelled.
    scoped = tuple(o for o in pf.open_orders if o.symbol in markets and o.symbol not in blocked)
    intents = reconcile(desired, scoped, markets=markets)
    flatten = flatten_intents(pf, markets, blocked, snapshot.ts)
    intents = _apply_global_cap(intents, snapshot, marks, equity, risk, targets)
    return Decision(tuple(flatten) + tuple(intents), halt)


def _keepable(open_orders, symbol, side, qty_held, target, band):
    """Our resting order for ``symbol`` that is still worth keeping, else ``None``."""
    best = None
    for o in open_orders:
        if o.symbol != symbol or o.side != side or not is_ours(o.client_id) or o.remaining <= 0:
            continue
        signed = o.remaining if side == "buy" else -o.remaining
        implied = (qty_held + signed) * o.price
        if abs(implied - target) <= 0.5 * band * target and (
            best is None or o.client_id < best.client_id
        ):
            best = o
    return best


@dataclass(frozen=True)
class RebalanceStrategy:
    """Picklable ``strategy`` for ``run_replay``; stateless (the state is passed through)."""

    params: RebalanceParams
    risk: RiskParams

    def __call__(self, snapshot: Snapshot, state):
        return decide_rebalance(snapshot, self.params, self.risk), state


__all__ = ["RebalanceParams", "RebalanceStrategy", "decide_rebalance"]
