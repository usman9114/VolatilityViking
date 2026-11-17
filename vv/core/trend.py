"""Trend + vol-targeting strategy maths (pure functions).

Implements research/g2_preregistration.json exactly. No I/O, no look-ahead:
every output at t depends only on inputs at or before t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MIN_HISTORY_DAYS = 120
_VOL_MIN_PERIODS = 30
_Z_WINDOW = 365
_Z_MIN_PERIODS = 30


@dataclass(frozen=True)
class TrendParams:
    lookbacks: tuple[int, ...] = (20, 60, 120)
    vol_halflife_days: float = 30.0
    portfolio_target_vol: float = 0.12
    per_coin_cap: float = 0.25
    gross_cap: float = 1.0
    allow_short: bool = False
    rebalance_min: float = 0.02  # band: max(rebalance_min, rebalance_frac * |target|)
    rebalance_frac: float = 0.25
    dial_floor: float = 0.25


def trend_scores(daily_close: pd.DataFrame) -> pd.DataFrame:
    """Mean over lookbacks of sign(close_t / close_{t-L} - 1); NaN until 120 days of history."""
    lookbacks = TrendParams().lookbacks
    signs = [np.sign(daily_close / daily_close.shift(lb) - 1.0) for lb in lookbacks]
    total = sum(signs)
    valid = signs[0].notna()
    for s in signs[1:]:
        valid &= s.notna()
    return (total / len(lookbacks)).where(valid)


def vol_forecast(daily_close: pd.DataFrame, halflife_days: float) -> pd.DataFrame:
    """EWMA std of daily log returns, annualised by sqrt(365)."""
    logret = np.log(daily_close).diff()
    vol = logret.ewm(halflife=halflife_days, min_periods=_VOL_MIN_PERIODS).std()
    return vol * np.sqrt(365.0)


def _trailing_z(s: pd.Series) -> pd.Series:
    roll = s.rolling(_Z_WINDOW, min_periods=_Z_MIN_PERIODS)
    std = roll.std().replace(0.0, np.nan)
    return (s - roll.mean()) / std


def risk_dial(features: pd.DataFrame, floor: float = 0.25) -> pd.Series:
    """m = 1 - 0.25*[z(stable_chg)<-1] - 0.25*[z(vix)>1] - 0.25*[z(ls)>1], clipped to [floor, 1].

    Missing feature column or NaN z-score contributes no penalty.
    """
    idx = features.index
    penalty = pd.Series(0.0, index=idx)
    rules = (
        ("stablecoin_supply_chg_30d", lambda z: z < -1.0),
        ("vix_level", lambda z: z > 1.0),
        ("global_ls_ratio", lambda z: z > 1.0),
    )
    for col, fire in rules:
        if col not in features.columns:
            continue
        z = _trailing_z(features[col].astype(float))
        hit = fire(z) & z.notna()
        penalty += 0.25 * hit.astype(float)
    return (1.0 - penalty).clip(lower=floor, upper=1.0)


def target_weights(
    daily_close: pd.DataFrame, params: TrendParams, dial: pd.Series | None = None
) -> pd.DataFrame:
    """Target weights decided at daily close t (applied by the simulator from t+1h)."""
    scores = trend_scores(daily_close)
    vol = vol_forecast(daily_close, params.vol_halflife_days)
    eligible = scores.notna() & vol.notna() & (vol > 0)
    n_elig = eligible.sum(axis=1).astype(float)
    scale = params.portfolio_target_vol / np.sqrt(n_elig.where(n_elig > 0))
    raw = scores.where(eligible).mul(scale, axis=0) / vol.where(eligible)
    w = raw.clip(-params.per_coin_cap, params.per_coin_cap)
    if not params.allow_short:
        w = w.clip(lower=0.0)
    w = w.fillna(0.0)
    gross = w.abs().sum(axis=1)
    factor = (params.gross_cap / gross.where(gross > 0)).clip(upper=1.0).fillna(1.0)
    w = w.mul(factor, axis=0)
    if dial is not None:
        w = w.mul(dial.reindex(w.index).fillna(1.0), axis=0)
    return w


def apply_band(current: pd.Series, target: pd.Series, params: TrendParams) -> pd.Series:
    """Per coin keep current unless |target - current| >= max(min, frac * |target|)."""
    cur = current.reindex(target.index).fillna(0.0)
    band = np.maximum(params.rebalance_min, params.rebalance_frac * target.abs())
    trade = (target - cur).abs() >= band - 1e-12
    return target.where(trade, cur)
