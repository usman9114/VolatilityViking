"""Synthetic data, registry and FakeVenue helpers for runner tests (no network)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from vv.core.types import SymbolFilters
from vv.exec.binance_perp import PerpAdapter
from vv.exec.fake import FakeVenue
from vv.runner.config import RunnerConfig, venue_symbol

SYMS = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
NOW = pd.Timestamp("2026-06-10 00:05", tz="UTC")
LAST = pd.Timestamp("2026-06-09", tz="UTC")
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0)


def synth_close(n: int = 320, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=LAST, periods=n, freq="D", tz="UTC", name="date")
    drift = {"AAAUSDT": 0.004, "BBBUSDT": 0.003, "CCCUSDT": -0.004, "DDDUSDT": -0.003}
    cols = {s: 100 * np.exp(np.cumsum(rng.normal(drift[s], 0.02, n))) for s in SYMS}
    return pd.DataFrame(cols, index=idx)


def make_fetcher(frame: pd.DataFrame, drop: tuple[str, pd.Timestamp] | None = None):
    calls = []

    def fetch(symbol, start, end):
        calls.append((symbol, start, end))
        df = pd.DataFrame({"close": frame[symbol], "volume": 1000.0})
        df = df[(df.index >= start) & (df.index < end)]
        if drop and drop[0] == symbol:
            df = df[df.index != drop[1]]
        return df

    fetch.calls = calls
    return fetch


def write_repo(tmp_path, t1=(3.0, 0.3), status="live", ids=("P1",)):
    """Registry + prereg under tmp_path/repo; returns the repo root."""
    root = tmp_path / "repo"
    (root / "research/signals").mkdir(parents=True)
    prereg = {"variants": {"P1": {"venue": "perp", "allow_short": True, "risk_dial": False},
                           "P2": {"venue": "perp", "allow_short": True, "risk_dial": True},
                           "S1": {"venue": "spot", "allow_short": False, "risk_dial": False}}}
    pp = root / "research/prereg.json"
    pp.write_text(json.dumps(prereg))
    sha = hashlib.sha256(pp.read_bytes()).hexdigest()
    sigs = [{"id": i, "family": "trend", "role": "directional", "hypothesis": "h", "sign": 1,
             "horizon": "1d", "data": "perp", "prereg_path": "research/prereg.json",
             "prereg_sha256": sha, "status": status, "created": "2026-01-01",
             "t1": {"sr_oos_stress": t1[0], "se": t1[1], "years": 2.0}} for i in ids]
    (root / "research/signals/registry.yaml").write_text(yaml.safe_dump({"signals": sigs}))
    return root


def make_cfg(tmp_path, root, **kw) -> RunnerConfig:
    base = dict(symbols=SYMS, venue="perp", repo_root=str(root), ledger_path="research/signals/none.csv",
                state_db=str(tmp_path / "state/runner.db"), data_dir=str(tmp_path / "data"),
                external_dir=str(tmp_path / "ext"), reports_dir=str(tmp_path / "reports"),
                ramp_base=1.0)
    base.update(kw)
    return RunnerConfig(**base)


def fake_venue(frame: pd.DataFrame, wallet: float = 10_000.0) -> FakeVenue:
    v = FakeVenue(mode="perp", wallet=wallet)
    for s in SYMS:
        vs = venue_symbol(s, "perp")
        px = float(frame[s].iloc[-1])
        v._filters[vs] = FILT
        v.quotes[vs] = (px * 0.9995, px * 1.0005)
    return v


def factory_for(venue: FakeVenue):
    made = []

    def factory(cfg, mode, halt_active):
        made.append(mode)
        return PerpAdapter(venue, mode=mode, halt_active=halt_active, sleep=lambda s: None)

    factory.made = made
    return factory


@pytest.fixture
def frame():
    return synth_close()
