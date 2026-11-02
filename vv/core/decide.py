"""The strategy entry point. Signature only; logic is ported in WP-S1.2."""

from __future__ import annotations

from vv.core.params import GridParams, RiskParams
from vv.core.types import OrderIntent, Snapshot


def decide(snapshot: Snapshot, params: GridParams, risk: RiskParams) -> list[OrderIntent]:
    """Map a portfolio-wide snapshot to order intents.

    Contract:
    - Pure: no I/O, no clock reads, no randomness, no mutation of inputs.
      Same inputs give the same output.
    - Returns ``[]`` if ``not snapshot.fresh`` or ``snapshot.portfolio.halt.active``.
    - Sees the whole portfolio (shared wallet), so exposure caps in ``risk``
      apply across all symbols, not per symbol in isolation.
    """
    raise NotImplementedError("strategy logic is ported in WP-S1.2")
