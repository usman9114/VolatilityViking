"""Deterministic client order ids."""

from __future__ import annotations

import hashlib

from vv.core.types import Side

PREFIX = "vv-"


def client_id(symbol: str, side: Side, level: int, epoch_bucket: int) -> str:
    """Stable id for (symbol, side, level, epoch_bucket): ``vv-`` + 32 hex = 35 chars.

    Uses blake2b, not ``hash()``, so it is identical across processes and
    restarts. Fits Binance's ``newClientOrderId`` (``[A-Za-z0-9_-]{1,36}``).
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"invalid side: {side!r}")
    if level < 0 or epoch_bucket < 0:
        raise ValueError("level and epoch_bucket must be >= 0")
    key = f"{symbol}|{side}|{level}|{epoch_bucket}".encode()
    return PREFIX + hashlib.blake2b(key, digest_size=16).hexdigest()
