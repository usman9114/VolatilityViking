# Changelog

## [Unreleased] - 2025-12-26

### Added
- **Neutral Liquidity Strategy (Scalping Mode)**: Optimized for low-volatility "chop" markets.
    - Grid Span tightened to `0.6%` (0.006).
    - Trend Bias disabled (`0.0`) to center orders on price.
- **Dynamic Safety Factor**:
    - Automatically widens grid in high-ADX trends (Safety Factor scales from 2.5x to 4.0x).
    - Prevents scalping configuration from becoming dangerous during crashes.
- **Aggressive Override**:
    - `aggressive_mode: true` allows manual grid settings to bypass GARCH safety minimums.
- **Monitoring Tool**: `tools/monitor.py` for real-time health checks of bot logs.
- **Dynamic Capital Allocation**:
    - **Core Tiers**: `ETH`, `SOL` receive higher wallet exposure (e.g., 50%).
    - **Forager Tiers**: Auto-discovered symbols (`LINK`, `AVAX`) receive lower exposure (e.g., 15%).

### Changed
- **Order Management Logic**: 
    - Implemented **"Drift Tolerance"** (configurable, default 0.2%) to prevent order churn.
    - Fixed **"Stuck Orders"** bug by implementing a "Mark & Sweep" cleanup phase (cancels orphaned orders).
- **Configuration**:
    - `grid_settings` moved to `config.json` with new parameters (`drift_tolerance`).
    - `wallet_exposure_limit` split into `core_wallet_exposure` and `forager_wallet_exposure`.

### Fixed
- **Crash Fix**: Added missing `cancel_order` helper method to `AsyncMultiSymbolBot` to resolve `AttributeError`.
- **Logic Fix**: Corrected GARCH trend scaling double-application bug in backtester.
