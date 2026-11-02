"""Interfaces (ports) the live loop, FakeExchange and replay implement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from vv.core.types import (
    CancelOrder,
    HaltState,
    OpenOrder,
    PlaceOrder,
    SymbolFilters,
    SymbolMarket,
)

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class ExchangeAdapter(Protocol):
    async def load_markets(self) -> dict[str, SymbolFilters]:
        """Trading filters per symbol (also lets the adapter attach them to tickers)."""
        ...

    async def fetch_tickers(self, symbols: Sequence[str]) -> dict[str, SymbolMarket]: ...

    async def fetch_balance(self) -> dict[str, float]:
        """Free balance per asset, e.g. ``{"USDT": 812.3, "BTC": 0.01}``."""
        ...

    async def fetch_open_orders(self, symbol: str | None = None) -> list[OpenOrder]: ...

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int, since: datetime | None = None
    ) -> pd.DataFrame:
        """Closed bars, oldest first, columns open/high/low/close/volume, UTC DatetimeIndex."""
        ...

    async def create_order(self, order: PlaceOrder) -> OpenOrder: ...

    async def cancel_order(self, order: CancelOrder) -> None: ...

    async def fetch_my_trades(
        self, symbol: str, since: datetime | None = None
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Timezone-aware UTC time."""
        ...


@runtime_checkable
class StateStore(Protocol):
    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any) -> None: ...

    def record_order(self, order: OpenOrder) -> None: ...

    def record_fill(self, fill: Mapping[str, Any]) -> None: ...

    def load_halt(self) -> HaltState: ...

    def save_halt(self, halt: HaltState) -> None: ...
