"""Core domain types. Frozen dataclasses; plain floats (no Decimal) in the core.

Money-exact rounding to exchange filters lives in ``vv.core.filters``.
All ``exposure`` / ``equity`` values are quote-currency notionals (USDT), not
fractions of equity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pandas is only needed for annotations here
    import pandas as pd

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class SymbolFilters:
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float


@dataclass(frozen=True)
class SymbolMarket:
    symbol: str
    ts: datetime
    bid: float
    ask: float
    last: float
    filters: SymbolFilters


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_entry: float
    realized_pnl: float


@dataclass(frozen=True)
class OpenOrder:
    symbol: str
    side: Side
    price: float
    qty: float
    filled: float
    client_id: str
    exchange_id: str
    ts: datetime

    @property
    def remaining(self) -> float:
        return self.qty - self.filled


@dataclass(frozen=True)
class HaltState:
    active: bool = False
    reason: str | None = None
    since: datetime | None = None


@dataclass(frozen=True)
class PortfolioState:
    """One shared wallet across all coins.

    ``cash_free`` is spendable quote currency; ``cash_reserved`` is quote
    currency locked by resting buy orders. Both are owned, so both count
    towards equity.
    """

    ts: datetime
    cash_free: float
    cash_reserved: float
    positions: dict[str, Position]
    open_orders: tuple[OpenOrder, ...]
    peak_equity: float
    halt: HaltState

    def equity(self, marks: Mapping[str, float]) -> float:
        """Free + reserved cash + mark-to-market value of all positions.

        Raises ``KeyError`` if a non-zero position has no mark.
        """
        held = sum(p.qty * marks[p.symbol] for p in self.positions.values() if p.qty)
        return self.cash_free + self.cash_reserved + held

    def exposure_by_symbol(self, marks: Mapping[str, float]) -> dict[str, float]:
        """Per symbol: holdings at mark + notional of the unfilled part of resting buys."""
        out: dict[str, float] = {}
        for p in self.positions.values():
            if p.qty:
                out[p.symbol] = out.get(p.symbol, 0.0) + p.qty * marks[p.symbol]
        for o in self.open_orders:
            if o.side == "buy":
                out[o.symbol] = out.get(o.symbol, 0.0) + o.price * o.remaining
        return out

    def exposure(self, marks: Mapping[str, float]) -> float:
        """Total of ``exposure_by_symbol``: holdings plus resting buy notional."""
        return sum(self.exposure_by_symbol(marks).values())


@dataclass(frozen=True)
class SymbolFeatures:
    """Latest per-symbol strategy inputs (see ``vv.core.features``)."""

    close: float
    sigma_h: float  # horizon volatility as a fraction (log-return std over the horizon)
    ema_4h: float
    regime_ok: bool  # close >= ema_4h
    warm: bool  # enough history for every field; if False, do not trade on it


@dataclass(frozen=True)
class Snapshot:
    ts: datetime
    markets: dict[str, SymbolMarket]
    bars: dict[str, pd.DataFrame]
    portfolio: PortfolioState
    stale: tuple[str, ...] = ()
    features: dict[str, SymbolFeatures] = field(default_factory=dict)

    @property
    def fresh(self) -> bool:
        return not self.stale


@dataclass(frozen=True)
class PlaceOrder:
    symbol: str
    side: Side
    price: float
    qty: float
    client_id: str
    post_only: bool = True


@dataclass(frozen=True)
class CancelOrder:
    symbol: str
    client_id: str


OrderIntent = PlaceOrder | CancelOrder
