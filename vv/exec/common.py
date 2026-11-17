"""Shared types and the target-position rebalancer used by perp, spot and fake venues."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from vv.core.filters import round_price, round_qty, validate_notional
from vv.core.ids import PREFIX
from vv.core.types import Side, SymbolFilters

SECRET_ENV_VARS = ("VV_BINANCE_KEY", "VV_BINANCE_SECRET")
_SIG_RE = re.compile(r"(signature=)[0-9A-Fa-f]{16,}")


class SecretRedactingFilter(logging.Filter):
    """Scrubs API key/secret values (from env or given explicitly) from log records."""

    def __init__(self, extra: Sequence[str] = ()) -> None:
        super().__init__()
        self._extra = [s for s in extra if s]

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = [os.environ.get(n, "") for n in SECRET_ENV_VARS] + self._extra
        msg = record.getMessage()
        for s in secrets:
            if s and len(s) >= 4:
                msg = msg.replace(s, "***")
        record.msg = _SIG_RE.sub(r"\1***", msg)
        record.args = ()
        return True


LOG = logging.getLogger("vv.exec")
LOG.addFilter(SecretRedactingFilter())


class TradingHalted(RuntimeError):
    pass


class StartupCheckFailed(RuntimeError):
    pass


class DuplicateOrder(Exception):
    """Venue rejected a clientOrderId it has already seen."""


class OrderRejected(Exception):
    pass


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    qty: float  # signed coin quantity
    entry: float
    mark: float
    liq_price: float | None
    adl_rank: int | None


@dataclass(frozen=True)
class Balance:
    wallet: float
    available: float
    margin: float  # equity: wallet + unrealized pnl


@dataclass(frozen=True)
class OrderResult:
    symbol: str
    side: Side
    qty: float
    price: float | None
    client_id: str
    status: str  # dry_run|new|filled|partially_filled|canceled|expired|rejected|skipped
    filled: float = 0.0
    reduce_only: bool = False
    stage: str = "maker"  # maker|ioc
    note: str = ""


class VenueClient(Protocol):
    """Minimal synchronous venue surface the rebalancer needs."""

    def filters(self, symbols: Sequence[str]) -> dict[str, SymbolFilters]: ...
    def quote(self, symbol: str) -> tuple[float, float]: ...
    def positions(self) -> dict[str, PositionInfo]: ...
    def balance(self) -> Balance: ...
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
    ) -> OrderResult: ...
    def get_order(self, symbol: str, client_id: str) -> OrderResult: ...
    def cancel(self, symbol: str, client_id: str) -> None: ...


def client_order_id(symbol: str, side: Side, qty: float, decision_ts: int, stage: str = "maker") -> str:
    """Deterministic ``vv-`` + 32 hex id: same decision => same id (idempotent retries)."""
    key = f"{symbol}|{side}|{qty!r}|{decision_ts}|{stage}".encode()
    return PREFIX + hashlib.blake2b(key, digest_size=16).hexdigest()


def clip_targets(
    targets: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    *,
    gross_cap: float = 1.0,
    max_short_frac: float = 0.5,
    allow_short: bool = True,
) -> dict[str, float]:
    """Scale targets so sum|notional| <= gross_cap*equity and short notional <= max_short_frac*equity.

    Only ever scales magnitudes down (factor <= 1); signs are preserved.
    """
    out = {s: float(q) for s, q in targets.items()}
    if not allow_short:
        out = {s: max(q, 0.0) for s, q in out.items()}
    eq = max(equity, 0.0)
    short = sum(-q * prices[s] for s, q in out.items() if q < 0)
    short_cap = max_short_frac * eq
    if short > short_cap:
        f = short_cap / short
        out = {s: (q * f if q < 0 else q) for s, q in out.items()}
    gross = sum(abs(q) * prices[s] for s, q in out.items())
    gross_lim = gross_cap * eq
    if gross > gross_lim:
        f = gross_lim / gross
        out = {s: q * f for s, q in out.items()}
    return out


class Rebalancer:
    allow_short = True
    supports_reduce_only = True

    def __init__(
        self,
        client: VenueClient,
        *,
        halt_active: Callable[[], bool],
        sleep: Callable[[float], None] = time.sleep,
        fill_wait_s: float = 15.0,
        slippage: float = 0.002,
    ) -> None:
        self.client = client
        self.halt_active = halt_active
        self._sleep = sleep
        self.fill_wait_s = fill_wait_s
        self.slippage = slippage
        self.ready = False

    # reads -----------------------------------------------------------------
    def positions(self) -> dict[str, PositionInfo]:
        return self.client.positions()

    def balance(self) -> Balance:
        return self.client.balance()

    def _guard(self) -> None:
        if self.halt_active():
            raise TradingHalted("halt active: refusing to trade")
        if not self.ready:
            raise StartupCheckFailed("startup checks have not passed: refusing to trade")

    # rebalance --------------------------------------------------------------
    def rebalance(
        self,
        targets: dict[str, float],
        *,
        max_short_frac: float = 0.5,
        gross_cap: float = 1.0,
        dry_run: bool = False,
        decision_ts: int,
    ) -> list[OrderResult]:
        self._guard()
        symbols = list(targets)
        flt = self.client.filters(symbols)
        quotes = {s: self.client.quote(s) for s in symbols}
        mid = {s: (b + a) / 2 for s, (b, a) in quotes.items()}
        equity = self.client.balance().margin
        pos = self.client.positions()
        clipped = clip_targets(
            targets,
            mid,
            equity,
            gross_cap=gross_cap,
            max_short_frac=max_short_frac,
            allow_short=self.allow_short,
        )
        reduces: list[tuple[str, Side, float]] = []
        entries: list[tuple[str, Side, float, bool]] = []  # (.., needs_reduce_first)
        for s in symbols:
            cur = pos[s].qty if s in pos else 0.0
            delta = clipped[s] - cur
            if delta == 0:
                continue
            side: Side = "buy" if delta > 0 else "sell"
            reduce_qty = min(abs(delta), abs(cur)) if cur * delta < 0 else 0.0
            entry_qty = abs(delta) - reduce_qty
            step = flt[s].step_size
            rq, eq_ = round_qty(reduce_qty, step), round_qty(entry_qty, step)
            px = mid[s]
            flip = reduce_qty > 0 and entry_qty > 0
            if rq > 0 and validate_notional(px, rq, flt[s]):
                reduces.append((s, side, rq))
            elif flip:
                continue  # cannot complete the reduce leg: never open the other side
            if eq_ > 0 and validate_notional(px, eq_, flt[s]):
                entries.append((s, side, eq_, flip))
        results: list[OrderResult] = []
        reduced_ok: dict[str, bool] = {}
        for s, side, q in reduces:
            res = self._execute(s, side, q, self.supports_reduce_only, decision_ts, flt[s], dry_run)
            results += res
            reduced_ok[s] = dry_run or sum(r.filled for r in res) >= q - 1e-12
        for s, side, q, flip in entries:
            if flip and not reduced_ok.get(s, False):
                continue
            results += self._execute(s, side, q, False, decision_ts, flt[s], dry_run)
        return results

    def _execute(
        self,
        symbol: str,
        side: Side,
        qty: float,
        reduce_only: bool,
        ts: int,
        f: SymbolFilters,
        dry_run: bool,
    ) -> list[OrderResult]:
        if self.halt_active():
            raise TradingHalted("halt active: refusing to trade")
        bid, ask = self.client.quote(symbol)
        maker_px = round_price(side, bid if side == "buy" else ask, f.tick_size)
        cid = client_order_id(symbol, side, qty, ts, "maker")
        if dry_run:
            return [OrderResult(symbol, side, qty, maker_px, cid, "dry_run", 0.0, reduce_only)]
        out: list[OrderResult] = []
        first = self._place(symbol, side, qty, maker_px, "GTX", reduce_only, cid, "maker")
        out.append(first)
        filled = first.filled
        if first.status not in ("filled", "rejected") and filled < qty:
            self._sleep(self.fill_wait_s)
            cur = self.client.get_order(symbol, cid)
            filled = cur.filled
            out[-1] = cur
            if cur.status not in ("filled", "canceled", "expired", "rejected"):
                self.client.cancel(symbol, cid)
                out[-1] = self.client.get_order(symbol, cid)
                filled = out[-1].filled
        remaining = round_qty(qty - filled, f.step_size)
        if remaining > 0 and filled < qty:
            bid, ask = self.client.quote(symbol)
            px = ask * (1 + self.slippage) if side == "buy" else bid * (1 - self.slippage)
            px = round_price(side, px, f.tick_size)
            if validate_notional(px, remaining, f):
                icid = client_order_id(symbol, side, remaining, ts, "ioc")
                out.append(self._place(symbol, side, remaining, px, "IOC", reduce_only, icid, "ioc"))
        return out

    def _place(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        tif: str,
        reduce_only: bool,
        cid: str,
        stage: str,
    ) -> OrderResult:
        try:
            r = self.client.place(
                symbol, side, qty, price, tif=tif, reduce_only=reduce_only, client_id=cid
            )
        except DuplicateOrder:
            LOG.warning("duplicate client id %s for %s: treating as already submitted", cid, symbol)
            r = self.client.get_order(symbol, cid)
        except OrderRejected as e:
            r = OrderResult(symbol, side, qty, price, cid, "rejected", 0.0, reduce_only, note=str(e))
        return OrderResult(
            r.symbol, r.side, r.qty, r.price, r.client_id, r.status, r.filled, reduce_only, stage, r.note
        )
