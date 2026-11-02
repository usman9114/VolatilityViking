"""Position sizing: inverse-vol per-symbol budgets and geometric ladder quantities. Pure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from vv.core.params import RiskParams

_SIGMA_FLOOR = 1e-6  # keeps 1/sigma finite for a degenerate (zero-vol) estimate


def symbol_budgets(
    equity: float, sigmas: Mapping[str, float], risk: RiskParams
) -> dict[str, float]:
    """Quote-currency exposure budget per symbol.

    ``total_exposure_cap * equity`` is split in proportion to ``1 / sigma`` (calmer coins get
    more), then each share is capped at ``per_symbol_cap * equity``. Capped excess is not
    redistributed, so the budgets always sum to at most the total cap.
    """
    if not sigmas or equity <= 0:
        return {s: 0.0 for s in sigmas}
    inv = {s: 1.0 / max(float(v), _SIGMA_FLOOR) for s, v in sigmas.items()}
    total_inv = sum(inv.values())
    pool = risk.total_exposure_cap * equity
    cap = risk.per_symbol_cap * equity
    return {s: min(pool * w / total_inv, cap) for s, w in inv.items()}


def ladder_quantities(
    prices: Sequence[float], budget_remaining: float, qty_multiplier: float
) -> list[float]:
    """Quantity per ladder level (same order as ``prices``).

    Notional weights are geometric (``qty_multiplier ** i``), so deeper levels get more when
    the multiplier is > 1. The notional of the whole ladder sums to ``budget_remaining``
    (before the caller rounds quantities down, which can only reduce it). A budget that is
    zero or negative yields all-zero quantities: never a negative size.
    """
    n = len(prices)
    budget = max(0.0, float(budget_remaining))
    if n == 0:
        return []
    if budget == 0.0:
        return [0.0] * n
    weights = [qty_multiplier**i for i in range(n)]
    total_w = sum(weights)
    return [budget * w / total_w / p for w, p in zip(weights, prices, strict=True)]


def budget_remaining(budget: float, exposure: float) -> float:
    """Budget left after current exposure (holdings at mark + resting buys). Never negative."""
    return max(0.0, budget - exposure)
