"""The strategy entry point: a pure map from a portfolio-wide snapshot to a ``Decision``."""

from __future__ import annotations

from vv.core import filters as flt
from vv.core.orders import anchor_bucket, is_ours, order_id, reconcile
from vv.core.params import GridParams, RiskParams
from vv.core.pricing import buy_ladder, step_size, take_profit_price
from vv.core.regime import buys_allowed
from vv.core.risk import evaluate_halt, flatten_intents
from vv.core.sizing import budget_remaining, ladder_quantities, symbol_budgets
from vv.core.types import (
    CancelOrder,
    Decision,
    OrderIntent,
    PlaceOrder,
    Snapshot,
)


def _marks(snapshot: Snapshot) -> dict[str, float]:
    """Mid prices; a position with no market falls back to its cost basis (neutral mark)."""
    marks = {s: (m.bid + m.ask) / 2.0 for s, m in snapshot.markets.items()}
    for p in snapshot.portfolio.positions.values():
        marks.setdefault(p.symbol, p.avg_entry)
    return marks


def decide(snapshot: Snapshot, params: GridParams, risk: RiskParams) -> Decision:
    """Map a portfolio-wide snapshot to order intents plus the halt state to persist.

    Pure: no I/O, no clock reads, no randomness, no mutation of inputs.

    - Stale data: no intents at all, halt unchanged (a halt condition is not acted on from
      data we do not trust).
    - Active halt: cancel everything and flatten.
    - Otherwise per warm, unblocked symbol: a buy ladder (below the bid, sized within the
      per-symbol budget net of holdings) and one take-profit sell; reconciled against resting
      orders; new buys are finally clipped by the cumulative global exposure cap.
    """
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
    warm = [
        s for s in universe if (f := snapshot.features.get(s)) is not None and f.warm
    ]

    sigmas = {s: snapshot.features[s].sigma_h * params.vol_scalar.get(s, 1.0) for s in warm}
    budgets = symbol_budgets(equity, sigmas, risk)

    desired: list[PlaceOrder] = []
    for s in warm:
        m, feat = markets[s], snapshot.features[s]
        flt_ = m.filters
        pos = pf.positions.get(s)
        held = pos is not None and pos.qty > 0
        avg_entry = pos.avg_entry if held else None
        step = step_size(feat, params, s)

        if buys_allowed(feat):
            # The ladder is the *target* resting set, so our own resting buys are part of it
            # and must not be subtracted here (that would cancel the ladder as it fills the
            # budget). Holdings and foreign resting buys do consume budget; the final
            # cumulative cap below counts everything that will actually be resting.
            committed = (pos.qty * marks[s] if held and pos is not None else 0.0) + sum(
                o.price * o.remaining
                for o in pf.open_orders
                if o.symbol == s and o.side == "buy" and not is_ours(o.client_id)
            )
            remaining = budget_remaining(budgets.get(s, 0.0), committed)
            prices = [
                flt.round_price("buy", p, flt_.tick_size)
                for p in buy_ladder(m.bid, avg_entry, step, params.levels)
            ]
            qtys = ladder_quantities(prices, remaining, params.qty_multiplier)
            anchor = m.bid if avg_entry is None else min(m.bid, avg_entry)
            bucket = anchor_bucket(anchor, step)
            for level, (p, q) in enumerate(zip(prices, qtys, strict=True)):
                q = flt.round_qty(q, flt_.step_size)
                if q > 0:
                    cid = order_id(s, "buy", level, p, q, bucket)
                    desired.append(PlaceOrder(s, "buy", p, q, cid))

        if held and pos is not None:
            foreign_sells = sum(
                o.remaining
                for o in pf.open_orders
                if o.symbol == s and o.side == "sell" and not is_ours(o.client_id)
            )
            free = flt.round_qty(max(0.0, pos.qty - foreign_sells), flt_.step_size)
            tp = flt.round_price(
                "sell", take_profit_price(pos.avg_entry, step, params, m.ask), flt_.tick_size
            )
            if free > 0:
                bucket = anchor_bucket(pos.avg_entry, step)
                cid = order_id(s, "sell", 0, tp, free, bucket)
                desired.append(PlaceOrder(s, "sell", tp, free, cid))

    # Orders in scope for reconcile: everything of warm symbols; for symbols we cannot
    # evaluate (cold or no features) only our resting buys, which are cancelled.
    warm_set = set(warm)
    scoped = tuple(
        o
        for o in pf.open_orders
        if o.symbol in warm_set
        or (o.symbol in markets and o.symbol not in blocked and o.side == "buy")
    )
    intents = reconcile(desired, scoped, markets=markets)

    # Blocked symbols: flatten (not part of the ladder logic at all).
    flatten = flatten_intents(pf, markets, blocked, snapshot.ts)

    intents = _apply_global_cap(intents, snapshot, marks, equity, risk, budgets)
    return Decision(tuple(flatten) + tuple(intents), halt)


def _apply_global_cap(
    intents: list[OrderIntent],
    snapshot: Snapshot,
    marks: dict[str, float],
    equity: float,
    risk: RiskParams,
    budgets: dict[str, float],
) -> list[OrderIntent]:
    """Drop new buys that would push running exposure past the total or per-symbol cap.

    Running exposure starts from holdings plus every resting buy that survives this cycle
    (cancelled ones no longer count), then each new buy is added in turn. Symbols are
    visited by budget descending (ties by name); within a symbol, shallowest level first,
    and the first buy that does not fit drops the rest of that symbol's ladder.
    """
    pf = snapshot.portfolio
    cancelled = {i.client_id for i in intents if isinstance(i, CancelOrder)}
    per_sym = dict(pf.exposure_by_symbol(marks))
    for o in pf.open_orders:
        if o.side == "buy" and o.client_id in cancelled:
            per_sym[o.symbol] = per_sym.get(o.symbol, 0.0) - o.price * o.remaining
    total = sum(per_sym.values())
    cap_total = risk.total_exposure_cap * equity

    new_buys: dict[str, list[PlaceOrder]] = {}
    for i in intents:
        if isinstance(i, PlaceOrder) and i.side == "buy":
            new_buys.setdefault(i.symbol, []).append(i)

    allowed: set[int] = set()
    for s in sorted(new_buys, key=lambda x: (-budgets.get(x, 0.0), x)):
        cap_sym = budgets.get(s, 0.0)
        for o in sorted(new_buys[s], key=lambda x: -x.price):
            notional = o.price * o.qty
            if (
                total + notional > cap_total + 1e-9
                or per_sym.get(s, 0.0) + notional > cap_sym + 1e-9
            ):
                break
            total += notional
            per_sym[s] = per_sym.get(s, 0.0) + notional
            allowed.add(id(o))

    return [
        i
        for i in intents
        if not (isinstance(i, PlaceOrder) and i.side == "buy") or id(i) in allowed
    ]
