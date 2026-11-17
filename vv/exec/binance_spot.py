"""Binance spot adapter: long-only, no margin. Same interface as PerpAdapter."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from vv.core.types import Side, SymbolFilters
from vv.exec.common import (
    Balance,
    DuplicateOrder,
    OrderRejected,
    OrderResult,
    PositionInfo,
    Rebalancer,
    VenueClient,
)


def _f(x: Any) -> float:
    return float(x) if x not in (None, "") else 0.0


class CcxtSpotClient:
    """ccxt ``binance`` spot wrapper (untested live here; fake-tested via the same Rebalancer)."""

    def __init__(self, quote: str = "USDT", *, exchange: Any = None) -> None:
        if exchange is None:
            import ccxt

            cfg: dict[str, Any] = {"enableRateLimit": True,
                                   "options": {"adjustForTimeDifference": True}}
            key, secret = os.environ.get("VV_BINANCE_KEY"), os.environ.get("VV_BINANCE_SECRET")
            if key and secret:
                cfg["apiKey"], cfg["secret"] = key, secret
            exchange = ccxt.binance(cfg)
        self.ex, self.quote_ccy = exchange, quote

    def __repr__(self) -> str:
        return "CcxtSpotClient()"

    def filters(self, symbols: Sequence[str]) -> dict[str, SymbolFilters]:
        if not self.ex.markets:
            self.ex.load_markets()
        out = {}
        for s in symbols:
            m = self.ex.market(s)
            out[s] = SymbolFilters(_f(m["precision"]["price"]), _f(m["precision"]["amount"]),
                                   _f((m["limits"]["amount"] or {}).get("min")),
                                   _f((m["limits"]["cost"] or {}).get("min")) or 5.0)
        return out

    def quote(self, symbol: str) -> tuple[float, float]:
        ob = self.ex.fetch_order_book(symbol, 5)
        return float(ob["bids"][0][0]), float(ob["asks"][0][0])

    def positions(self) -> dict[str, PositionInfo]:
        bal = self.ex.fetch_balance()
        out = {}
        for ccy, amt in bal.get("total", {}).items():
            if ccy == self.quote_ccy or not amt:
                continue
            sym = f"{ccy}/{self.quote_ccy}"
            if sym in self.ex.markets:
                b, a = self.quote(sym)
                out[sym] = PositionInfo(sym, float(amt), 0.0, (b + a) / 2, None, None)
        return out

    def balance(self) -> Balance:
        bal = self.ex.fetch_balance()
        free = _f(bal.get("free", {}).get(self.quote_ccy))
        held = sum(p.qty * p.mark for p in self.positions().values())
        total = _f(bal.get("total", {}).get(self.quote_ccy))
        return Balance(total, free, total + held)

    def place(self, symbol: str, side: Side, qty: float, price: float, *, tif: str,
              reduce_only: bool, client_id: str) -> OrderResult:
        params = {"timeInForce": tif if tif != "GTX" else "GTC", "newClientOrderId": client_id}
        otype = "limit_maker" if tif == "GTX" else "limit"
        if otype == "limit_maker":
            params.pop("timeInForce")
            params["postOnly"] = True
            otype = "limit"
        try:
            o = self.ex.create_order(symbol, otype, side, qty, price, params)
        except Exception as e:
            if "uplicate" in str(e) or "-2010" in str(e) and "client" in str(e).lower():
                raise DuplicateOrder(client_id) from None
            if "-2010" in str(e):
                raise OrderRejected(str(e)[:200]) from None
            raise
        return self._res(o, symbol)

    @staticmethod
    def _res(o: dict[str, Any], symbol: str) -> OrderResult:
        st = {"open": "new", "closed": "filled", "canceled": "canceled",
              "expired": "expired", "rejected": "rejected"}.get(o.get("status"), "new")
        return OrderResult(symbol, o["side"], _f(o.get("amount")), _f(o.get("price")) or None,
                           o.get("clientOrderId") or "", st, _f(o.get("filled")))

    def get_order(self, symbol: str, client_id: str) -> OrderResult:
        return self._res(self.ex.fetch_order(None, symbol, {"origClientOrderId": client_id}), symbol)

    def cancel(self, symbol: str, client_id: str) -> None:
        self.ex.cancel_order(None, symbol, {"origClientOrderId": client_id})


class SpotAdapter(Rebalancer):
    """Long-only, cash-only: negative targets are clipped to 0; no reduceOnly on spot."""

    allow_short = False
    supports_reduce_only = False

    def __init__(self, client: VenueClient | None = None, *,
                 halt_active: Callable[[], bool], **kw: Any) -> None:
        super().__init__(client or CcxtSpotClient(), halt_active=halt_active, **kw)

    def start(self, symbols: Sequence[str] = ()) -> None:
        sync = getattr(self.client, "sync_time", None)
        if sync:
            sync()
        self.ready = True
