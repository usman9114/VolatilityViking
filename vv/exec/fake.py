"""In-memory venue (perp or spot) for tests and dry-run runs. No network."""

from __future__ import annotations

from collections.abc import Sequence

from vv.core.types import Side, SymbolFilters
from vv.exec.common import (
    Balance,
    DuplicateOrder,
    OrderRejected,
    OrderResult,
    PositionInfo,
)


class FakeVenue:
    def __init__(
        self,
        *,
        mode: str = "perp",
        wallet: float = 1000.0,
        filters: dict[str, SymbolFilters] | None = None,
        quotes: dict[str, tuple[float, float]] | None = None,
        slippage: float = 0.0005,
        mmr: float = 0.004,
        postonly_fills: bool = True,
        hedge_mode: bool = False,
    ) -> None:
        self.mode = mode
        self.wallet = wallet
        self._filters = dict(filters or {})
        self.quotes = dict(quotes or {})
        self.slippage = slippage
        self.mmr = mmr
        self.postonly_fills = postonly_fills
        self.hedge_mode = hedge_mode
        self.margin_mode: dict[str, str] = {}
        self.leverage: dict[str, int] = {}
        self.qty: dict[str, float] = {}
        self.entry: dict[str, float] = {}
        self.orders: dict[str, OrderResult] = {}
        self.placed: list[OrderResult] = []  # every accepted place() call, in order
        self.time_synced = False

    # setup helpers ----------------------------------------------------------
    def set_quote(self, symbol: str, bid: float, ask: float) -> None:
        self.quotes[symbol] = (bid, ask)

    def sync_time(self) -> None:
        self.time_synced = True

    def _mid(self, s: str) -> float:
        b, a = self.quotes[s]
        return (b + a) / 2

    def liq_price(self, symbol: str) -> float | None:
        q = self.qty.get(symbol, 0.0)
        if not q or self.mode == "spot":
            return None
        lev = self.leverage.get(symbol, 1)
        e = self.entry[symbol]
        return e * (1 - 1 / lev + self.mmr) if q > 0 else e * (1 + 1 / lev - self.mmr)

    def apply_funding(self, rate: float) -> None:
        """Longs pay ``rate`` x notional when positive; shorts receive."""
        for s, q in self.qty.items():
            self.wallet -= q * self._mid(s) * rate

    # VenueClient ------------------------------------------------------------
    def filters(self, symbols: Sequence[str]) -> dict[str, SymbolFilters]:
        return {s: self._filters[s] for s in symbols}

    def quote(self, symbol: str) -> tuple[float, float]:
        return self.quotes[symbol]

    def positions(self) -> dict[str, PositionInfo]:
        return {
            s: PositionInfo(s, q, self.entry[s], self._mid(s), self.liq_price(s), None)
            for s, q in self.qty.items()
            if q
        }

    def balance(self) -> Balance:
        unreal = sum((self._mid(s) - self.entry[s]) * q for s, q in self.qty.items() if q)
        if self.mode == "spot":
            held = sum(q * self._mid(s) for s, q in self.qty.items())
            return Balance(self.wallet, self.wallet, self.wallet + held)
        margin = self.wallet + unreal
        used = sum(abs(q) * self._mid(s) / self.leverage.get(s, 1) for s, q in self.qty.items())
        return Balance(self.wallet, max(margin - used, 0.0), margin)

    def place(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        *,
        tif: str,
        reduce_only: bool,
        client_id: str,
    ) -> OrderResult:
        if client_id in self.orders:
            raise DuplicateOrder(client_id)
        cur = self.qty.get(symbol, 0.0)
        signed = qty if side == "buy" else -qty
        if reduce_only and (cur == 0 or cur * signed > 0 or abs(signed) > abs(cur) + 1e-12):
            raise OrderRejected("ReduceOnly order would increase or flip position")
        if self.mode == "spot" and (reduce_only or cur + signed < -1e-12):
            raise OrderRejected("spot: no shorting / reduceOnly")
        bid, ask = self.quotes[symbol]
        res = OrderResult(symbol, side, qty, price, client_id, "new", 0.0, reduce_only)
        if tif == "GTX":
            crosses = price >= ask if side == "buy" else price <= bid
            if crosses:
                res = OrderResult(symbol, side, qty, price, client_id, "rejected", 0, reduce_only,
                                  note="post-only would cross")
            elif self.postonly_fills:
                self._fill(symbol, signed, price)
                res = OrderResult(symbol, side, qty, price, client_id, "filled", qty, reduce_only)
        else:  # IOC: marketable at quote +/- slippage if within the limit
            fill_px = ask * (1 + self.slippage) if side == "buy" else bid * (1 - self.slippage)
            ok = price >= fill_px if side == "buy" else price <= fill_px
            if ok:
                self._fill(symbol, signed, fill_px)
                res = OrderResult(symbol, side, qty, fill_px, client_id, "filled", qty, reduce_only)
            else:
                res = OrderResult(symbol, side, qty, price, client_id, "expired", 0, reduce_only)
        self.orders[client_id] = res
        self.placed.append(res)
        return res

    def _fill(self, symbol: str, signed: float, px: float) -> None:
        cur = self.qty.get(symbol, 0.0)
        new = cur + signed
        if self.mode == "spot":
            self.wallet -= signed * px
        elif cur and cur * signed < 0:
            closed = min(abs(signed), abs(cur)) * (1 if cur > 0 else -1)
            self.wallet += closed * (px - self.entry[symbol])
        if cur == 0 or cur * signed > 0:
            tot = abs(cur) + abs(signed)
            self.entry[symbol] = (abs(cur) * self.entry.get(symbol, px) + abs(signed) * px) / tot
        elif abs(signed) > abs(cur):
            self.entry[symbol] = px
        self.qty[symbol] = 0.0 if abs(new) < 1e-12 else new

    def get_order(self, symbol: str, client_id: str) -> OrderResult:
        return self.orders[client_id]

    def cancel(self, symbol: str, client_id: str) -> None:
        o = self.orders[client_id]
        if o.status == "new":
            self.orders[client_id] = OrderResult(
                o.symbol, o.side, o.qty, o.price, o.client_id, "canceled", o.filled, o.reduce_only
            )

    # mode surface used by PerpAdapter startup checks -------------------------
    def get_hedge_mode(self) -> bool:
        return self.hedge_mode

    def set_hedge_mode(self, on: bool) -> None:
        self.hedge_mode = on

    def get_margin_mode(self, symbol: str) -> str:
        return self.margin_mode.get(symbol, "cross")

    def set_margin_mode(self, symbol: str, mode: str) -> None:
        self.margin_mode[symbol] = mode

    def get_leverage(self, symbol: str) -> int:
        return self.leverage.get(symbol, 20)

    def set_leverage(self, symbol: str, lev: int) -> None:
        self.leverage[symbol] = lev
