"""Sandbox helper for the ops drills. Offline: writes a runner config and synthetic daily bars.

  seed.py make DIR [--drop-yesterday] [--halt REASON]   # writes DIR/runner.yaml and DIR/data
  seed.py orders DIR        # prints number of order rows in DIR/state/runner.db
  seed.py halted DIR        # prints 1 if an active halt exists, else 0
  seed.py runs DIR          # prints number of run rows recorded

Synthetic prices only. Nothing here touches an exchange or a real key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vv.runner.data import DailyStore, expected_last_bar  # noqa: E402
from vv.runner.state import State  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]
HISTORY_DAYS = 420


def make(sandbox: Path, drop_yesterday: bool, halt: str | None) -> None:
    sandbox.mkdir(parents=True, exist_ok=True)
    cfg = {
        "symbols": SYMBOLS, "venue": "spot", "repo_root": str(REPO),
        "registry_path": "research/signals/registry.yaml", "ledger_path": "research/signals/ledger.csv",
        "state_db": str(sandbox / "state/runner.db"), "data_dir": str(sandbox / "data"),
        "external_dir": str(sandbox / "external"), "reports_dir": str(sandbox / "reports"),
        "mode": "dry", "history_days": HISTORY_DAYS,
    }
    (sandbox / "runner.yaml").write_text(yaml.safe_dump(cfg))
    now = pd.Timestamp.now(tz="UTC")
    last = expected_last_bar(now)
    idx = pd.date_range(end=last, periods=HISTORY_DAYS, freq="D", tz="UTC", name="date")
    rng = np.random.default_rng(7)
    rows = []
    for s in SYMBOLS:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(idx))))
        rows.append(pd.DataFrame({"symbol": s, "date": idx, "close": close, "volume": 1000.0,
                                  "ingested_at": now}))
    frame = pd.concat(rows, ignore_index=True)
    if drop_yesterday:
        frame = frame[frame["date"] != last]
    DailyStore(sandbox / "data").append(frame)
    if halt:
        st = State(sandbox / "state/runner.db")
        st.set_halt(halt)
        st.close()


def _state(sandbox: Path) -> State:
    return State(sandbox / "state/runner.db")


def main(argv: list[str]) -> int:
    cmd, sandbox = argv[1], Path(argv[2])
    if cmd == "make":
        drop = "--drop-yesterday" in argv
        halt = argv[argv.index("--halt") + 1] if "--halt" in argv else None
        make(sandbox, drop, halt)
    elif cmd == "orders":
        st = _state(sandbox)
        print(st.db.execute("SELECT COUNT(*) FROM orders").fetchone()[0])
        st.close()
    elif cmd == "runs":
        st = _state(sandbox)
        print(st.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
        st.close()
    elif cmd == "halted":
        st = _state(sandbox)
        print(1 if st.halt_active() else 0)
        st.close()
    else:
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
