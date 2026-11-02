"""Order diffing: turn desired orders plus open orders into cancel/place intents. Pure.

Only orders whose client id starts with ``vv-`` are ever cancelled; anything else on the
account is somebody else's and is invisible here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence

from vv.core import filters as flt
from vv.core.ids import PREFIX
from vv.core.types import (
    CancelOrder,
    OpenOrder,
    OrderIntent,
    PlaceOrder,
    Side,
    SymbolMarket,
)

QTY_TOLERANCE = 0.10


def is_ours(client_id: str) -> bool:
    return client_id.startswith(PREFIX)


def anchor_bucket(anchor: float, step: float) -> int:
    """Epoch bucket for client ids: the ladder anchor's index on a geometric grid of ``step``.

    Constant while the anchor stays inside one step-wide band, so ids are stable across
    restarts and only change when the ladder genuinely re-anchors. Always >= 0.
    """
    if anchor <= 0 or step <= 0:
        return 0
    idx = math.floor(math.log(anchor) / math.log1p(step))
    return max(0, idx + 100_000)


def order_id(
    symbol: str, side: Side, level: int, price: float, qty: float, epoch_bucket: int
) -> str:
    """Client id unique per placement yet deterministic: ``vv-`` + 32 hex (35 chars).

    Exchanges never let a client id be reused, so the id covers the *rounded* price and qty
    as well as symbol/side/level/epoch: an order that is resized or repriced gets a new id,
    while an identical order always hashes to the same one (stable across restarts).
    Pass price and qty already rounded to the exchange filters.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"invalid side: {side!r}")
    key = f"{symbol}|{side}|{level}|{price!r}|{qty!r}|{epoch_bucket}".encode()
    return PREFIX + hashlib.blake2b(key, digest_size=16).hexdigest()


def normalise(order: PlaceOrder, market: SymbolMarket | None) -> PlaceOrder | None:
    """Round to filters and apply the safety guards; ``None`` means drop the order."""
    if market is None:
        return None
    f = market.filters
    price = flt.round_price(order.side, order.price, f.tick_size)
    qty = flt.round_qty(order.qty, f.step_size)
    if price <= 0 or not flt.validate_notional(price, qty, f):
        return None
    if order.post_only:
        if order.side == "buy" and price >= market.bid:
            return None
        if order.side == "sell" and price <= market.bid:
            return None
    if (price, qty) == (order.price, order.qty):
        return order
    return PlaceOrder(order.symbol, order.side, price, qty, order.client_id, order.post_only)


def _matches(order: OpenOrder, want: PlaceOrder, tick: float, tolerance_ticks: int) -> bool:
    if order.symbol != want.symbol or order.side != want.side:
        return False
    if abs(order.price - want.price) > tolerance_ticks * tick + 1e-12 * max(1.0, want.price):
        return False
    return abs(order.remaining - want.qty) <= QTY_TOLERANCE * want.qty


def reconcile(
    desired: Iterable[PlaceOrder],
    open_orders: Sequence[OpenOrder],
    tolerance_ticks: int = 2,
    *,
    markets: Mapping[str, SymbolMarket],
) -> list[OrderIntent]:
    """Cancels first (in input order), then places (in ``desired`` order).

    ``markets`` supplies tick size, filters and bid for rounding and guards; the brief's
    two-argument form cannot express those.
    """
    wanted = [w for d in desired if (w := normalise(d, markets.get(d.symbol))) is not None]
    ours = [o for o in open_orders if is_ours(o.client_id) and o.remaining > 0]

    kept: set[int] = set()
    unmatched: list[PlaceOrder] = []
    for w in wanted:
        tick = markets[w.symbol].filters.tick_size
        cands = [
            (abs(o.price - w.price), abs(o.remaining - w.qty), i)
            for i, o in enumerate(ours)
            if i not in kept and _matches(o, w, tick, tolerance_ticks)
        ]
        if cands:
            kept.add(min(cands)[2])
        else:
            unmatched.append(w)

    kept_ids = {ours[i].client_id for i in kept}
    out: list[OrderIntent] = [
        CancelOrder(o.symbol, o.client_id) for i, o in enumerate(ours) if i not in kept
    ]
    # A new order may not reuse the id of one we are keeping (the exchange would reject it).
    out.extend(w for w in unmatched if w.client_id not in kept_ids)
    return out
