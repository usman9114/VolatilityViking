"""Binance USD-M perpetuals adapter (ccxt ``binanceusdm``): one-way, isolated, 1x."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from vv.core.types import Side, SymbolFilters
from vv.exec.common import (
    LOG,
    Balance,
    DuplicateOrder,
    OrderRejected,
    OrderResult,
    PositionInfo,
    Rebalancer,
    StartupCheckFailed,
    VenueClient,
)

TESTNET_URL = "https://testnet.binancefuture.com"
_STATUS = {
    "NEW": "new", "FILLED": "filled", "PARTIALLY_FILLED": "partially_filled",
    "CANCELED": "canceled", "EXPIRED": "expired", "REJECTED": "rejected",
}


def _f(x: Any) -> float:
    return float(x) if x not in (None, "") else 0.0


class CcxtPerpClient:
    """Thin ccxt wrapper. Keys come from env only and are never logged or repr'd."""

    def __init__(self, mode: str = "testnet", *, exchange: Any = None) -> None:
        if mode not in ("testnet", "live"):
            raise ValueError("mode must be 'testnet' or 'live'")
        self.mode = mode
        if exchange is None:
            import ccxt

            cfg: dict[str, Any] = {
                "enableRateLimit": True,
                "options": {"adjustForTimeDifference": True, "defaultType": "future"},
            }
            key, secret = os.environ.get("VV_BINANCE_KEY"), os.environ.get("VV_BINANCE_SECRET")
            if key and secret:
                cfg["apiKey"], cfg["secret"] = key, secret
            proxy = os.environ.get("HTTPS_PROXY")
            if proxy:  # ccxt does not pick the env proxy up on its own
                cfg["https_proxy"] = proxy
                cfg["verify"] = os.environ.get("SSL_CERT_FILE") or True
            exchange = ccxt.binanceusdm(cfg)
            if mode == "testnet":
                # VV_BINANCE_TEST_VENUE: "demo" (default; Binance Demo Trading, demo-fapi.binance.com) or
                # "testnet" (legacy testnet.binancefuture.com). Keys only work on the
                # venue they were created on.
                venue = os.environ.get("VV_BINANCE_TEST_VENUE", "demo").strip().lower()
                if venue == "demo":
                    exchange.enable_demo_trading(True)
                elif venue == "testnet":
                    exchange.set_sandbox_mode(True)
                else:
                    raise ValueError("VV_BINANCE_TEST_VENUE must be 'testnet' or 'demo'")
        self.ex = exchange

    def __repr__(self) -> str:
        return f"CcxtPerpClient(mode={self.mode!r})"

    def sync_time(self) -> None:
        self.ex.load_time_difference()

    def filters(self, symbols: Sequence[str]) -> dict[str, SymbolFilters]:
        if not self.ex.markets:
            self.ex.load_markets()
        out = {}
        for s in symbols:
            m = self.ex.market(s)
            out[s] = SymbolFilters(
                tick_size=_f(m["precision"]["price"]),
                step_size=_f(m["precision"]["amount"]),
                min_qty=_f((m["limits"]["amount"] or {}).get("min")),
                min_notional=_f((m["limits"]["cost"] or {}).get("min")) or 5.0,
            )
        return out

    def quote(self, symbol: str) -> tuple[float, float]:
        ob = self.ex.fetch_order_book(symbol, 5)
        return float(ob["bids"][0][0]), float(ob["asks"][0][0])

    def _risk(self) -> list[dict[str, Any]]:
        return self.ex.fapiPrivateV2GetPositionRisk()

    def positions(self) -> dict[str, PositionInfo]:
        adl: dict[str, int] = {}
        try:
            for r in self.ex.fapiPrivateGetAdlQuantile():
                q = r.get("adlQuantile", {})
                adl[r["symbol"]] = int(q.get("BOTH", 0))
        except Exception as e:  # ADL rank is informational
            LOG.warning("adl quantile unavailable: %s", type(e).__name__)
        out = {}
        for r in self._risk():
            amt = _f(r.get("positionAmt"))
            if not amt:
                continue
            sym = self.ex.safe_symbol(r["symbol"], None, None, "swap") if hasattr(
                self.ex, "safe_symbol") else r["symbol"]
            liq = _f(r.get("liquidationPrice")) or None
            out[sym] = PositionInfo(sym, amt, _f(r.get("entryPrice")), _f(r.get("markPrice")),
                                    liq, adl.get(r["symbol"]))
            LOG.info("position %s qty=%s liq=%s adl=%s", sym, amt, liq, adl.get(r["symbol"]))
        return out

    def balance(self) -> Balance:
        a = self.ex.fapiPrivateV2GetAccount()
        return Balance(_f(a["totalWalletBalance"]), _f(a["availableBalance"]),
                       _f(a["totalMarginBalance"]))

    def _risk_for(self, symbol: str) -> dict[str, Any]:
        mid = self.ex.market(symbol)["id"]
        rows = self.ex.fapiPrivateV2GetPositionRisk({"symbol": mid})
        return rows[0]

    def get_hedge_mode(self) -> bool:
        r = self.ex.fapiPrivateGetPositionSideDual()
        return str(r["dualSidePosition"]).lower() == "true"

    def set_hedge_mode(self, on: bool) -> None:
        self.ex.set_position_mode(on)

    def get_margin_mode(self, symbol: str) -> str:
        return str(self._risk_for(symbol)["marginType"]).lower()

    def set_margin_mode(self, symbol: str, mode: str) -> None:
        self.ex.set_margin_mode(mode, symbol)

    def get_leverage(self, symbol: str) -> int:
        return int(_f(self._risk_for(symbol)["leverage"]))

    def set_leverage(self, symbol: str, lev: int) -> None:
        self.ex.set_leverage(lev, symbol)

    @staticmethod
    def _to_result(o: dict[str, Any], symbol: str, reduce_only: bool) -> OrderResult:
        st = _STATUS.get(str(o.get("info", {}).get("status", o.get("status", ""))).upper(), "new")
        return OrderResult(symbol, o["side"], _f(o.get("amount")), _f(o.get("price")) or None,
                           o.get("clientOrderId") or "", st, _f(o.get("filled")), reduce_only)

    def place(self, symbol: str, side: Side, qty: float, price: float, *, tif: str,
              reduce_only: bool, client_id: str) -> OrderResult:
        params = {"timeInForce": tif, "newClientOrderId": client_id}
        if reduce_only:
            params["reduceOnly"] = True
        try:
            o = self.ex.create_order(symbol, "limit", side, qty, price, params)
        except Exception as e:
            msg = str(e)
            if "-4015" in msg or "uplicate" in msg:
                raise DuplicateOrder(client_id) from None
            if "-2022" in msg or "-5022" in msg or "-4164" in msg or "ReduceOnly" in msg:
                raise OrderRejected(type(e).__name__ + ": " + msg[:200]) from None
            raise
        return self._to_result(o, symbol, reduce_only)

    def get_order(self, symbol: str, client_id: str) -> OrderResult:
        o = self.ex.fetch_order(None, symbol, {"origClientOrderId": client_id})
        return self._to_result(o, symbol, bool(o.get("reduceOnly")))

    def cancel(self, symbol: str, client_id: str) -> None:
        self.ex.cancel_order(None, symbol, {"origClientOrderId": client_id})


class PerpAdapter(Rebalancer):
    """Target-position rebalancer for USD-M perps. Call ``start`` before trading."""

    allow_short = True
    supports_reduce_only = True

    def __init__(
        self,
        client: VenueClient | None = None,
        *,
        mode: str = "testnet",
        halt_active: Callable[[], bool],
        **kw: Any,
    ) -> None:
        super().__init__(client or CcxtPerpClient(mode), halt_active=halt_active, **kw)

    def start(self, symbols: Sequence[str]) -> None:
        """Verify time sync, one-way mode, isolated margin, leverage 1. Refuse on any failure."""
        self.ready = False
        c: Any = self.client
        try:
            c.sync_time()
            if c.get_hedge_mode():
                if c.positions():
                    raise StartupCheckFailed("hedge mode is on and positions exist")
                c.set_hedge_mode(False)
                if c.get_hedge_mode():
                    raise StartupCheckFailed("could not switch to one-way mode")
            held = c.positions()
            for s in symbols:
                if c.get_margin_mode(s) != "isolated":
                    if s in held:
                        raise StartupCheckFailed(f"{s}: open position on non-isolated margin")
                    c.set_margin_mode(s, "isolated")
                    if c.get_margin_mode(s) != "isolated":
                        raise StartupCheckFailed(f"{s}: could not set isolated margin")
                if c.get_leverage(s) != 1:
                    c.set_leverage(s, 1)
                    if c.get_leverage(s) != 1:
                        raise StartupCheckFailed(f"{s}: could not set leverage 1")
        except StartupCheckFailed:
            raise
        except Exception as e:
            raise StartupCheckFailed(f"startup check error: {type(e).__name__}") from e
        self.ready = True
        LOG.info("perp startup checks passed for %d symbols", len(symbols))
