# WP-T1 Trend + vol-targeting strategy and risk-off dial (pure functions)
Model: sonnet · Depends: feature screen (S-R2) · Worktree: no (only touch `vv/core/trend.py`, `tests/core/test_trend.py`) · Budget: 70k · Commit: NO

## Goal
Implement the strategy maths **exactly** as in `research/g2_preregistration.json`. Read it and don't change it. Another agent (WP-T2) is building the simulator and runner against the interface below, so **keep the signatures exactly**.

## Interface (`vv/core/trend.py`)
```python
@dataclass(frozen=True)
class TrendParams:
    lookbacks: tuple[int, ...] = (20, 60, 120)
    vol_halflife_days: float = 30.0
    portfolio_target_vol: float = 0.12
    per_coin_cap: float = 0.25
    gross_cap: float = 1.0
    allow_short: bool = False
    rebalance_min: float = 0.02      # band: max(rebalance_min, rebalance_frac * |target|)
    rebalance_frac: float = 0.25
    dial_floor: float = 0.25

def trend_scores(daily_close: pd.DataFrame) -> pd.DataFrame: ...
    # index: daily UTC; columns: symbols; value at t uses closes <= t only; NaN until 120 days of history
def vol_forecast(daily_close: pd.DataFrame, halflife_days: float) -> pd.DataFrame: ...   # annualised
def risk_dial(features: pd.DataFrame, floor: float = 0.25) -> pd.Series: ...
    # features columns: stablecoin_supply_chg_30d, vix_level, global_ls_ratio (already lagged; raw, not z-scored).
    # Computes trailing 365d z-scores internally; missing term -> 0 penalty; result in [floor, 1]
def target_weights(daily_close: pd.DataFrame, params: TrendParams, dial: pd.Series | None = None) -> pd.DataFrame: ...
    # Weights decided at close t (applied from t+1h by the simulator). Eligibility: 120 days of history.
    # raw = score * (target_vol / sqrt(N_eligible_t)) / vol; clip to +-per_coin_cap; if not allow_short clip below at 0;
    # if sum|w| > gross_cap scale down proportionally; then multiply by dial (if given).
def apply_band(current: pd.Series, target: pd.Series, params: TrendParams) -> pd.Series: ...
    # returns the weights to hold after this decision: per coin, keep current unless |target-current| >= band
```

## Tests first (`tests/core/test_trend.py`)
- **No look-ahead:** truncating the input at t leaves every output at or before t unchanged (scores, vol, weights, dial).
- **Scores:** a monotone up-series → +1, down → −1. Mixed lookbacks give ±1/3.
- **Caps:** per-coin ±0.25 and gross ≤ 1 always hold (property test, 200 random panels, fixed seed).
- **Spot clamp:** with `allow_short=False`, no weight is ever below 0.
- **Dial:** all three penalties fire → the floor (0.25). Missing data → no penalty. The output stays in [0.25, 1].
- **Band:** a small change is ignored and a big one is executed. Moving from 0 to any target ≥ 0.02 executes.
- **Eligibility:** a coin with fewer than 120 days of history gets weight 0.

## Done when
`.venv/bin/pytest -q tests/core/test_trend.py && .venv/bin/ruff check vv/core/trend.py tests/core/test_trend.py`, with `make test` still green.

## Reply (≤100 words)
status · test line · anything T2 must know
