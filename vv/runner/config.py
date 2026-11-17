"""Runner configuration (YAML)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class Caps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gross: float = 1.0
    short: float = 0.5
    per_coin: float = 0.25


class RunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbols: list[str]
    venue: Literal["spot", "perp"] = "spot"
    registry_path: str = "research/signals/registry.yaml"
    ledger_path: str = "research/signals/ledger.csv"
    repo_root: str = "."
    caps: Caps = Caps()
    ramp_base: float = 0.05
    factor_budget: dict[str, float] = {"risk_on": 1.0}
    state_db: str = "state/runner.db"
    data_dir: str = "data"
    external_dir: str = "data/external"
    reports_dir: str = "reports"
    mode: Literal["dry", "testnet", "live"] = "dry"
    dry_equity: float = 10_000.0
    history_days: int = 420
    external_max_age_days: int = 5
    shadow_cost_bps: float = 10.0
    min_order_notional: float = 10.0
    statuses: list[str] = ["shadow", "live"]


def load_config(path: str | Path) -> RunnerConfig:
    return RunnerConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def venue_symbol(sym: str, venue: str) -> str:
    """BTCUSDT -> BTC/USDT (spot) or BTC/USDT:USDT (USD-M perp), as ccxt names them."""
    base = sym[:-4] if sym.endswith("USDT") else sym
    return f"{base}/USDT" if venue == "spot" else f"{base}/USDT:USDT"
