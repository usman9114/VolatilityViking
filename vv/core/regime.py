"""Regime gate: new buys only when the feature set is warm and price is above the 4h EMA."""

from __future__ import annotations

from vv.core.types import SymbolFeatures


def buys_allowed(features: SymbolFeatures) -> bool:
    return bool(features.warm and features.regime_ok)
