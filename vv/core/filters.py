"""Exchange-filter rounding. Decimal arithmetic internally, floats at the edge.

Order of operations: round price and qty first, then ``validate_notional``.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from vv.core.types import Side, SymbolFilters


def _d(x: float) -> Decimal:
    # 12 significant digits: 0.1 -> Decimal("0.1"), and binary noise such as
    # 0.1 + 0.2 == 0.30000000000000004 collapses to Decimal("0.3") instead of
    # being treated as a real excess that rounds a whole tick the wrong way.
    # Exchange prices/qtys have far fewer digits, so no real value is altered.
    return Decimal(format(float(x), ".12g"))


def _snap(value: float, increment: float, rounding: str) -> float:
    inc = _d(increment)
    if inc <= 0:
        raise ValueError(f"increment must be > 0, got {increment}")
    steps = (_d(value) / inc).to_integral_value(rounding=rounding)
    return float(steps * inc)


def round_price(side: Side, price: float, tick: float) -> float:
    """Buys round down, sells round up, to a multiple of ``tick``."""
    if side == "buy":
        return _snap(price, tick, ROUND_FLOOR)
    if side == "sell":
        return _snap(price, tick, ROUND_CEILING)
    raise ValueError(f"invalid side: {side!r}")


def round_qty(qty: float, step: float) -> float:
    """Always rounds down to a multiple of ``step``."""
    return _snap(qty, step, ROUND_FLOOR)


def validate_notional(price: float, qty: float, filters: SymbolFilters) -> bool:
    """True if (already rounded) ``qty`` and ``price`` meet min_qty and min_notional."""
    q = _d(qty)
    if q <= 0 or q < _d(filters.min_qty):
        return False
    return _d(price) * q >= _d(filters.min_notional)
