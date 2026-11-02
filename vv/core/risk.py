"""Latched halts and flatten intents. Pure: the caller persists the returned ``HaltState``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

from vv.core import filters as flt
from vv.core.orders import is_ours, order_id
from vv.core.params import RiskParams
from vv.core.types import (
    CancelOrder,
    HaltState,
    OrderIntent,
    PlaceOrder,
    PortfolioState,
    SymbolMarket,
)

# Sell limit this far below the bid. Non-post-only orders may be IOC-limit and expire if the
# next open gaps beyond the limit, so the discount is wide; decide() retries every call anyway.
FLATTEN_DISCOUNT = 0.01
FLATTEN_LEVEL = 9999  # reserved client-id level for flatten sells (ladder levels are small)
_PRICE_TOL_TICKS = 2
_QTY_TOL = 0.10


def evaluate_halt(
    portfolio: PortfolioState, marks: Mapping[str, float], risk: RiskParams, now: datetime
) -> HaltState:
    """Next halt state. Latched: an active halt or a blocked symbol never clears here."""
    halt = portfolio.halt
    active, reason, since = halt.active, halt.reason, halt.since
    blocked = set(halt.blocked_symbols)

    if not active:
        equity = portfolio.equity(marks)
        floor = portfolio.peak_equity * (1.0 - risk.portfolio_halt_dd)
        if equity < floor:
            active = True
            reason = f"portfolio drawdown: equity {equity:.2f} < {floor:.2f}"
            since = now

    if not active:
        tripped = {
            p.symbol
            for p in portfolio.positions.values()
            if p.qty > 0
            and p.symbol not in blocked
            and marks[p.symbol] < p.avg_entry * (1.0 - risk.symbol_halt_dd)
        }
        if tripped:
            blocked |= tripped
            if reason is None:
                reason = "symbol drawdown: " + ", ".join(sorted(tripped))
                since = now

    return HaltState(
        active=active, reason=reason, since=since, blocked_symbols=frozenset(blocked)
    )


def flatten_intents(
    portfolio: PortfolioState,
    markets: Mapping[str, SymbolMarket],
    symbols: Iterable[str],
    now: datetime,
) -> list[OrderIntent]:
    """Cancel every ``vv-`` order for ``symbols`` and sell the free quantity of each.

    Pass every symbol for a portfolio halt, or just ``halt.blocked_symbols`` for a symbol
    halt. The sell is a plain limit (not post-only) at ``bid * (1 - 0.01)``. It is re-emitted
    on every call while free quantity remains (an IOC that expired is simply retried), with a
    fresh client id each minute (``now`` is the epoch bucket) because ids are never reusable.
    If one of our sells is still resting at a matching price and size it is kept, so a resting
    flatten does not churn cancel/replace.
    """
    minute = int(now.timestamp() // 60)
    out: list[OrderIntent] = []
    places: list[OrderIntent] = []
    for sym in sorted(set(symbols)):
        mine = [o for o in portfolio.open_orders if o.symbol == sym and is_ours(o.client_id)]
        pos = portfolio.positions.get(sym)
        market = markets.get(sym)
        want: PlaceOrder | None = None
        if pos is not None and pos.qty > 0 and market is not None:
            f = market.filters
            foreign_sells = sum(
                o.remaining
                for o in portfolio.open_orders
                if o.symbol == sym and o.side == "sell" and not is_ours(o.client_id)
            )
            qty = flt.round_qty(max(0.0, pos.qty - foreign_sells), f.step_size)
            price = flt.round_price("sell", market.bid * (1.0 - FLATTEN_DISCOUNT), f.tick_size)
            if price > 0 and flt.validate_notional(price, qty, f):
                want = PlaceOrder(
                    sym,
                    "sell",
                    price,
                    qty,
                    order_id(sym, "sell", FLATTEN_LEVEL, price, qty, minute),
                    post_only=False,
                )
        keep = None
        if want is not None and market is not None:
            tick = market.filters.tick_size
            for o in mine:
                if (
                    o.side == "sell"
                    and abs(o.price - want.price) <= _PRICE_TOL_TICKS * tick
                    and abs(o.remaining - want.qty) <= _QTY_TOL * want.qty
                ):
                    keep = o
                    break
        out.extend(CancelOrder(o.symbol, o.client_id) for o in mine if o is not keep)
        if want is not None and keep is None:
            places.append(want)
    return out + places
