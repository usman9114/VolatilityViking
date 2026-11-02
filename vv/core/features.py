"""Vectorised per-symbol features from 1m bars. Causal: row t depends only on data up to t.

A bar's information (its close) is treated as known at the *start of the next
bar*, so a 5m or 4h aggregate is only used from the first 1m row at or after its
end. In-progress aggregates are never used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vv.core.types import SymbolFeatures

SIGMA_BASE_MINUTES = 5
SIGMA_HALFLIFE_BARS = 288  # 1 day of 5m bars
_COLUMNS = ["close", "sigma_h", "ema_4h", "regime_ok", "warm"]


def _available_at(series: pd.Series, period: str, index: pd.Index) -> pd.Series:
    """Re-index an aggregate (labelled by bar start) onto ``index``, usable once the bar is complete."""
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(period)
    union = index.union(shifted.index)
    return shifted.reindex(union).ffill().reindex(index)


def compute_features(
    bars_1m: pd.DataFrame, horizon_minutes: int, ema_span_4h: int = 200
) -> pd.DataFrame:
    """Features for every 1m timestamp of ``bars_1m`` (UTC-indexed OHLCV, one symbol)."""
    idx = bars_1m.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("bars_1m must have a DatetimeIndex")
    if not idx.is_monotonic_increasing:
        raise ValueError("bars_1m index must be sorted ascending")
    close = bars_1m["close"].astype(float)

    # sigma: EWMA (zero-mean) variance of completed 5m log returns, scaled to the horizon.
    c5 = close.resample(f"{SIGMA_BASE_MINUTES}min").last().dropna()
    r5 = np.log(c5).diff()
    var5 = (r5**2).ewm(halflife=SIGMA_HALFLIFE_BARS, min_periods=SIGMA_HALFLIFE_BARS).mean()
    sigma5 = np.sqrt(var5)
    sigma_h = _available_at(sigma5, f"{SIGMA_BASE_MINUTES}min", idx) * np.sqrt(
        horizon_minutes / SIGMA_BASE_MINUTES
    )

    # 4h EMA over completed 4h closes only.
    c4 = close.resample("4h").last().dropna()
    ema4 = c4.ewm(span=ema_span_4h, adjust=False, min_periods=ema_span_4h).mean()
    ema_4h = _available_at(ema4, "4h", idx)

    warm = sigma_h.notna() & ema_4h.notna() & close.notna()
    out = pd.DataFrame(
        {
            "close": close,
            "sigma_h": sigma_h,
            "ema_4h": ema_4h,
            "regime_ok": (close >= ema_4h) & warm,
            "warm": warm,
        },
        index=idx,
    )
    return out[_COLUMNS]


def latest_features(
    bars_1m: pd.DataFrame, horizon_minutes: int, ema_span_4h: int = 200
) -> SymbolFeatures:
    """The last row of ``compute_features`` as a ``SymbolFeatures``."""
    row = compute_features(bars_1m, horizon_minutes, ema_span_4h).iloc[-1]
    return SymbolFeatures(
        close=float(row["close"]),
        sigma_h=float(row["sigma_h"]),
        ema_4h=float(row["ema_4h"]),
        regime_ok=bool(row["regime_ok"]),
        warm=bool(row["warm"]),
    )
