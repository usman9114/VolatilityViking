"""FakeExchange: several symbols, one shared USDT wallet, pessimistic fills.

Implements ``vv.core.ports.ExchangeAdapter`` over historical 1m bars on a
virtual clock. It is the backbone of the G1 replay, so every modelling choice
errs on the side of *fewer / worse* fills.

Conventions
-----------
* A bar's index label is its **open** time (Binance convention). The bar is
  *closed*, and therefore visible, once ``now >= label + 1m``. At virtual time
  ``t`` the newest visible bar is the one labelled ``t - 1m``.
* ``advance_to(t)`` matches every bar that closes in ``(old_now, t]``. An order
  placed at ``t`` is first matchable against the bar labelled ``t`` (which
  closes at ``t + 1m``); it can never fill against a bar that opened before it
  was placed.
* Typical replay loop: ``for t in timeline: ex.advance_to(t); snapshot; decide;
  place/cancel``.
* Symbols are ``BASE/USDT`` or ``BASEUSDT``; the only quote asset is USDT.

Matching rules
--------------
* Resting (post-only) buy fills only on **trade-through**: ``low < price - tick``;
  sell: ``high > price + tick``. Touching the price is not a fill. Fill price is
  the limit price (no price improvement, even if the bar opens through it),
  fee = ``maker_fee``.
* Intrabar path: O->L->H->C when ``close >= open`` (buys fill first, highest
  price first, then sells, lowest first), else O->H->L->C (sells first, then
  buys). Ties break by placement order.
* ``fill_haircut < 1``: each eligible maker fill happens with that probability
  (seeded RNG, one draw per eligible order per bar); otherwise it keeps resting.
* Non-post-only orders are taker, IOC-limit semantics: they fill in full at the
  open of the first bar after placement, +/- ``taker_slippage_bps`` (buy up,
  sell down, snapped to the tick against the trader), fee = ``taker_fee``. If
  that price is beyond the order's limit (buy above / sell below) the order
  expires unfilled (see ``expired``). To flatten reliably use a sell limit far
  below the market. Taker fills are never haircut.
* Post-only buy at ``>= ask`` or sell at ``<= bid`` is rejected, never taken.
  Synthetic quote: ``bid/ask = last close -/+ tick/2`` (a spread of one tick).

Simplifications (S1.4 must know)
--------------------------------
* No partial fills: an order fills entirely or not at all.
* No queue position: trade-through stands in for it. Volume is ignored.
* Buys reserve ``price*qty*(1+fee)`` from ``cash_free`` into ``cash_reserved``.
* Fees are charged in quote currency; base quantity received is not reduced.
* ``avg_entry`` is the volume-weighted buy cost including buy fees. Sells book
  ``net proceeds - avg_entry*qty`` as realized PnL and leave ``avg_entry``.
* Fill records are stamped with the *close* time of the bar they happened in
  (the moment they become observable), including taker fills at the open.
* ``client_id`` may never be reused, even after fill or cancel.
* A symbol with no closed bar yet ("not listed") rejects orders and is
  omitted from ``fetch_tickers``. Gaps inside a symbol's bars are not traded;
  its ticker keeps the last known close.
* No rate limits, no self-trade prevention, no balance dust.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from vv.core.filters import round_price, validate_notional
from vv.core.types import (
    CancelOrder,
    HaltState,
    OpenOrder,
    PlaceOrder,
    PortfolioState,
    Position,
    SymbolFilters,
    SymbolMarket,
)

QUOTE = "USDT"
BAR = pd.Timedelta(minutes=1)
_OHLCV = ["open", "high", "low", "close", "volume"]
_CASH_EPS = 1e-9
_QTY_EPS = 1e-12


class OrderRejected(Exception):
    """Raised by ``create_order`` / ``cancel_order``; ``reason`` is a stable code.

    Codes: duplicate_client_id, unknown_symbol, not_listed, invalid_order,
    not_on_tick, not_on_step, below_min_notional, post_only_cross,
    insufficient_cash, insufficient_base, unknown_order.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _utc(ts: datetime | pd.Timestamp) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware (UTC)")
    return t.tz_convert("UTC").to_pydatetime()


def _split_symbol(symbol: str) -> str:
    """Base asset of ``BASE/USDT`` or ``BASEUSDT``."""
    for suffix in (f"/{QUOTE}", QUOTE):
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            return symbol[: -len(suffix)]
    raise ValueError(f"symbol {symbol!r} is not quoted in {QUOTE}")


class VirtualClock:
    """Settable UTC clock. ``advance_to`` is forward-only and notifies listeners."""

    def __init__(self, start: datetime | pd.Timestamp) -> None:
        self._now = _utc(start)
        self._listeners: list[Callable[[datetime, datetime], None]] = []

    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime | pd.Timestamp) -> None:
        new = _utc(ts)
        if new < self._now:
            raise ValueError(f"clock cannot go backwards: {new} < {self._now}")
        old, self._now = self._now, new
        if new > old:
            for fn in self._listeners:
                fn(old, new)

    def subscribe(self, fn: Callable[[datetime, datetime], None]) -> None:
        self._listeners.append(fn)


@dataclass
class _Order:
    symbol: str
    side: str
    price: float
    qty: float
    client_id: str
    post_only: bool
    exchange_id: str
    placed_at: datetime
    seq: int
    reserved: float  # quote currency locked out of cash_free (buys only)


@dataclass
class _Pos:
    qty: float = 0.0
    avg_entry: float = 0.0
    realized: float = 0.0


class FakeExchange:
    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        filters: dict[str, SymbolFilters],
        start_cash: float,
        maker_fee: float = 0.00075,
        taker_fee: float = 0.001,
        taker_slippage_bps: float = 10,
        fill_haircut: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not bars:
            raise ValueError("bars must contain at least one symbol")
        if start_cash < 0:
            raise ValueError("start_cash must be >= 0")
        if not 0.0 <= fill_haircut <= 1.0:
            raise ValueError("fill_haircut must be in [0, 1]")
        missing = [s for s in bars if s not in filters]
        if missing:
            raise ValueError(f"no filters for symbols: {missing}")

        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.taker_slippage = taker_slippage_bps / 10_000.0
        self.fill_haircut = fill_haircut
        self._rng = random.Random(seed)
        self._filters = {s: filters[s] for s in bars}
        self._base = {s: _split_symbol(s) for s in bars}

        frames = {s: self._clean(s, df) for s, df in bars.items()}
        index = frames[next(iter(frames))].index
        for df in frames.values():
            index = index.union(df.index)
        self._index: pd.DatetimeIndex = index.sort_values()
        self._close_ns = self._index.asi8 + BAR.value  # when each bar becomes visible
        self._bars = {s: df.reindex(self._index) for s, df in frames.items()}
        self._arr = {
            s: {c: df[c].to_numpy(dtype=float) for c in _OHLCV} for s, df in self._bars.items()
        }
        self._close_ff = {s: df["close"].ffill().to_numpy(dtype=float) for s, df in self._bars.items()}

        self.clock = VirtualClock(self._index[0])
        self.clock.subscribe(self._on_advance)

        self.cash_free = float(start_cash)
        self._pos: dict[str, _Pos] = {}
        self._orders: dict[str, _Order] = {}
        self._used_ids: set[str] = set()
        self._seq = 0
        self.fills: list[dict[str, Any]] = []
        self.expired: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _clean(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise ValueError(f"{symbol}: bars need a tz-aware UTC DatetimeIndex")
        need = [c for c in _OHLCV[:4] if c not in df.columns]
        if need:
            raise ValueError(f"{symbol}: bars missing columns {need}")
        if not df.index.is_monotonic_increasing or not df.index.is_unique:
            raise ValueError(f"{symbol}: bar index must be strictly increasing")
        out = df.copy()
        out.index = out.index.tz_convert("UTC").as_unit("ns")  # asi8 maths below assumes ns
        if "volume" not in out.columns:
            out["volume"] = 0.0
        return out[_OHLCV].astype(float).dropna(subset=["open", "high", "low", "close"])

    # ------------------------------------------------------------------ time
    def advance_to(self, ts: datetime | pd.Timestamp) -> None:
        """Move the virtual clock forward, matching every bar that closes on the way."""
        self.clock.advance_to(ts)

    def _n_closed(self, at: datetime) -> int:
        """Number of bars (positions in the common index) closed at ``at``."""
        return int(np.searchsorted(self._close_ns, pd.Timestamp(at).value, side="right"))

    def _last_close(self, symbol: str) -> float | None:
        n = self._n_closed(self.clock.now())
        if n == 0:
            return None
        v = self._close_ff[symbol][n - 1]
        return None if math.isnan(v) else float(v)

    def marks(self) -> dict[str, float]:
        """Last known close per listed symbol (for ``PortfolioState.equity``)."""
        out = {}
        for s in self._bars:
            last = self._last_close(s)
            if last is not None:
                out[s] = last
        return out

    def equity(self) -> float:
        return self.portfolio_state(peak_equity=0.0).equity(self.marks())

    # ---------------------------------------------------------------- matching
    def _on_advance(self, old: datetime, new: datetime) -> None:
        for k in range(self._n_closed(old), self._n_closed(new)):
            for symbol in self._bars:
                self._match_bar(symbol, k)

    def _match_bar(self, symbol: str, k: int) -> None:
        a = self._arr[symbol]
        o, h, low, c = a["open"][k], a["high"][k], a["low"][k], a["close"][k]
        if math.isnan(o):
            return  # no bar for this symbol in this minute
        open_ts = self._index[k].to_pydatetime()
        close_ts = open_ts + BAR.to_pytimedelta()
        tick = self._filters[symbol].tick_size
        eps = tick * 1e-6

        orders = [x for x in self._orders.values() if x.symbol == symbol and x.placed_at <= open_ts]

        # 1) taker orders execute at the open
        for od in sorted((x for x in orders if not x.post_only), key=lambda x: x.seq):
            self._fill_taker(od, o, tick, close_ts)

        # 2) resting orders trade through along the intrabar path
        buys = sorted(
            (x for x in orders if x.post_only and x.side == "buy" and low < x.price - tick - eps),
            key=lambda x: (-x.price, x.seq),
        )
        sells = sorted(
            (x for x in orders if x.post_only and x.side == "sell" and h > x.price + tick + eps),
            key=lambda x: (x.price, x.seq),
        )
        for od in (buys + sells) if c >= o else (sells + buys):
            if self.fill_haircut < 1.0 and self._rng.random() >= self.fill_haircut:
                continue
            self._execute(od, od.price, self.maker_fee, close_ts, "maker")

    def _fill_taker(self, od: _Order, open_px: float, tick: float, ts: datetime) -> None:
        # Slippage always against the trader, snapped to the tick against the trader too:
        # round_price rounds buys down / sells up, so use the opposite side to round up/down.
        if od.side == "buy":
            px = round_price("sell", open_px * (1 + self.taker_slippage), tick)
            ok = px <= od.price + tick * 1e-6
        else:
            px = round_price("buy", open_px * (1 - self.taker_slippage), tick)
            ok = px >= od.price - tick * 1e-6
        if not ok:
            self._release(od)
            self.expired.append(
                {"ts": ts, "symbol": od.symbol, "side": od.side, "client_id": od.client_id,
                 "limit": od.price, "would_fill": px}
            )
            return
        self._execute(od, px, self.taker_fee, ts, "taker")

    def _release(self, od: _Order) -> None:
        self.cash_free += od.reserved
        del self._orders[od.client_id]

    def _execute(self, od: _Order, price: float, fee_rate: float, ts: datetime, liq: str) -> None:
        notional = price * od.qty
        fee = notional * fee_rate
        pos = self._pos.setdefault(od.symbol, _Pos())
        del self._orders[od.client_id]
        if od.side == "buy":
            cost = notional + fee
            self.cash_free += od.reserved - cost
            total = pos.qty * pos.avg_entry + cost
            pos.qty = round(pos.qty + od.qty, 12)
            pos.avg_entry = total / pos.qty
        else:
            self.cash_free += notional - fee
            pos.realized += (notional - fee) - pos.avg_entry * od.qty
            pos.qty = round(pos.qty - od.qty, 12)
            if pos.qty <= _QTY_EPS:
                pos.qty, pos.avg_entry = 0.0, 0.0
        self.fills.append(
            {"ts": ts, "symbol": od.symbol, "side": od.side, "price": price, "qty": od.qty,
             "fee": fee, "client_id": od.client_id, "liquidity": liq}
        )

    # ------------------------------------------------------------ adapter port
    async def load_markets(self) -> dict[str, SymbolFilters]:
        return dict(self._filters)

    async def fetch_tickers(self, symbols: Sequence[str]) -> dict[str, SymbolMarket]:
        out: dict[str, SymbolMarket] = {}
        now = self.clock.now()
        for s in symbols:
            if s not in self._filters:
                raise ValueError(f"unknown symbol {s!r}")
            last = self._last_close(s)
            if last is None:
                continue  # not listed yet
            f = self._filters[s]
            out[s] = SymbolMarket(
                symbol=s, ts=now, bid=last - f.tick_size / 2, ask=last + f.tick_size / 2,
                last=last, filters=f,
            )
        return out

    async def fetch_balance(self) -> dict[str, float]:
        out = {QUOTE: self.cash_free}
        for s, p in self._pos.items():
            free = self._free_base(s)
            if free > _QTY_EPS:
                out[self._base[s]] = free
        return out

    async def fetch_open_orders(self, symbol: str | None = None) -> list[OpenOrder]:
        return [self._as_open(o) for o in self._orders.values() if symbol in (None, o.symbol)]

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int, since: datetime | None = None
    ) -> pd.DataFrame:
        if symbol not in self._bars:
            raise ValueError(f"unknown symbol {symbol!r}")
        tf = pd.Timedelta(timeframe)
        if tf < BAR or tf % BAR:
            raise ValueError(f"timeframe must be a whole number of minutes, got {timeframe!r}")
        ratio = tf // BAR
        now = self.clock.now()
        n = self._n_closed(now)
        df = self._bars[symbol]
        if since is None:
            lo, hi = max(0, n - (limit + 1) * ratio), n
        else:
            start = pd.Timestamp(_utc(since)).ceil(tf)
            lo = min(int(self._index.searchsorted(start, side="left")), n)
            hi = min(n, lo + (limit + 1) * ratio)
        sub = df.iloc[lo:hi]
        if ratio > 1 and len(sub):
            sub = sub.resample(tf, label="left", closed="left", origin="epoch").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            )
            if since is None and lo > 0 and sub.index[0] < df.index[lo]:
                sub = sub.iloc[1:]  # partial leading bucket
        sub = sub.dropna(subset=["close"])
        sub = sub[sub.index + tf <= pd.Timestamp(now)]  # completed buckets only
        sub = sub.tail(limit) if since is None else sub.head(limit)
        assert sub.empty or sub.index[-1] + tf <= pd.Timestamp(now), "look-ahead in fetch_ohlcv"
        return sub.copy()

    async def create_order(self, order: PlaceOrder) -> OpenOrder:
        self._validate(order)
        reserve = 0.0
        if order.side == "buy":
            fee = self.maker_fee if order.post_only else self.taker_fee
            reserve = order.price * order.qty * (1 + fee)
            if reserve > self.cash_free + _CASH_EPS:
                raise OrderRejected("insufficient_cash", f"need {reserve:.8f}, free {self.cash_free:.8f}")
        elif order.qty > self._free_base(order.symbol) + _QTY_EPS:
            raise OrderRejected("insufficient_base", f"{order.symbol} qty {order.qty}")
        self.cash_free -= reserve
        self._seq += 1
        od = _Order(
            symbol=order.symbol, side=order.side, price=order.price, qty=order.qty,
            client_id=order.client_id, post_only=order.post_only, exchange_id=str(self._seq),
            placed_at=self.clock.now(), seq=self._seq, reserved=reserve,
        )
        self._orders[od.client_id] = od
        self._used_ids.add(od.client_id)
        return self._as_open(od)

    async def cancel_order(self, order: CancelOrder) -> None:
        od = self._orders.get(order.client_id)
        if od is None or od.symbol != order.symbol:
            raise OrderRejected("unknown_order", f"{order.symbol} {order.client_id}")
        self._release(od)

    async def fetch_my_trades(
        self, symbol: str, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        lim = None if since is None else _utc(since)
        return [dict(f) for f in self.fills if f["symbol"] == symbol and (lim is None or f["ts"] >= lim)]

    # ----------------------------------------------------------------- helpers
    def _validate(self, o: PlaceOrder) -> None:
        if o.client_id in self._used_ids:
            raise OrderRejected("duplicate_client_id", o.client_id)
        if o.symbol not in self._filters:
            raise OrderRejected("unknown_symbol", o.symbol)
        last = self._last_close(o.symbol)
        if last is None:
            raise OrderRejected("not_listed", o.symbol)
        if o.side not in ("buy", "sell"):
            raise OrderRejected("invalid_order", f"side {o.side!r}")
        if not (math.isfinite(o.price) and math.isfinite(o.qty) and o.price > 0 and o.qty > 0):
            raise OrderRejected("invalid_order", f"price {o.price} qty {o.qty}")
        f = self._filters[o.symbol]
        if not math.isclose(round_price(o.side, o.price, f.tick_size), o.price, rel_tol=1e-9):
            raise OrderRejected("not_on_tick", f"{o.price} tick {f.tick_size}")
        steps = o.qty / f.step_size
        if not math.isclose(steps, round(steps), rel_tol=1e-9, abs_tol=1e-9):
            raise OrderRejected("not_on_step", f"{o.qty} step {f.step_size}")
        if not validate_notional(o.price, o.qty, f):
            raise OrderRejected("below_min_notional", f"{o.price} x {o.qty}")
        if o.post_only:
            bid, ask = last - f.tick_size / 2, last + f.tick_size / 2
            if (o.side == "buy" and o.price >= ask) or (o.side == "sell" and o.price <= bid):
                raise OrderRejected("post_only_cross", f"{o.side} {o.price} bid {bid} ask {ask}")

    def _free_base(self, symbol: str) -> float:
        held = self._pos[symbol].qty if symbol in self._pos else 0.0
        pending = sum(x.qty for x in self._orders.values() if x.symbol == symbol and x.side == "sell")
        return held - pending

    def _as_open(self, o: _Order) -> OpenOrder:
        return OpenOrder(
            symbol=o.symbol, side=o.side, price=o.price, qty=o.qty, filled=0.0,  # type: ignore[arg-type]
            client_id=o.client_id, exchange_id=o.exchange_id, ts=o.placed_at,
        )

    @property
    def cash_reserved(self) -> float:
        return sum(o.reserved for o in self._orders.values())

    def portfolio_state(
        self, peak_equity: float, halt: HaltState | None = None
    ) -> PortfolioState:
        positions = {
            s: Position(symbol=s, qty=p.qty, avg_entry=p.avg_entry, realized_pnl=p.realized)
            for s, p in self._pos.items()
            if p.qty or p.realized
        }
        return PortfolioState(
            ts=self.clock.now(),
            cash_free=self.cash_free,
            cash_reserved=self.cash_reserved,
            positions=positions,
            open_orders=tuple(self._as_open(o) for o in self._orders.values()),
            peak_equity=peak_equity,
            halt=halt if halt is not None else HaltState(),
        )


__all__ = ["FakeExchange", "OrderRejected", "VirtualClock"]
