"""Grid pricing maths. Pure; prices are not tick-rounded here (see ``vv.core.filters``)."""

from __future__ import annotations

from vv.core.params import GridParams
from vv.core.types import SymbolFeatures


def step_size(features: SymbolFeatures, params: GridParams, symbol: str) -> float:
    """Fractional grid step: the largest of the min-step, fee and volatility floors."""
    return max(
        params.min_step,
        params.tp_fee_multiple * params.round_trip_fee,
        params.spacing_sigmas * features.sigma_h * params.vol_scalar.get(symbol, 1.0),
    )


def buy_ladder(bid: float, avg_entry: float | None, step: float, levels: int) -> list[float]:
    """Buy prices below the reference, strictly decreasing and all below ``bid``.

    The reference is ``bid`` when flat, else ``min(bid, avg_entry)`` so that adds
    only happen below cost basis.
    """
    ref = bid if avg_entry is None else min(bid, avg_entry)
    factor = 1.0 - step
    return [ref * factor**i for i in range(1, levels + 1)]


def take_profit_price(avg_entry: float, step: float, params: GridParams, ask: float) -> float:
    """Sell price: at least one step and the fee multiple above cost basis, and never below ask."""
    return max(
        avg_entry * (1.0 + step),
        avg_entry * (1.0 + params.tp_fee_multiple * params.round_trip_fee),
        ask,
    )
