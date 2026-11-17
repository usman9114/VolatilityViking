"""CLI: python -m vv.runner --config C --mode {dry|testnet|live} | resume --reason "..." --config C"""
from __future__ import annotations

import argparse
import logging
import sys

from vv.runner.alerts import Alerter, redact
from vv.runner.config import load_config
from vv.runner.core import run
from vv.runner.state import State


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vv.runner")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("resume", help="clear a persisted halt")
    r.add_argument("--reason", required=True)
    r.add_argument("--config", default="config/runner.yaml")
    ap.add_argument("--config", default="config/runner.yaml")
    ap.add_argument("--mode", choices=["dry", "testnet", "live"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(a.config)
    if a.cmd == "resume":
        st = State(cfg.state_db)
        n = st.resume(a.reason)
        st.close()
        logging.getLogger("vv.runner").warning("halt resumed (%d cleared): %s", n, redact(a.reason))
        Alerter().alert(f"halt cleared by operator: {a.reason}")
        return 0
    return run(cfg, a.mode).code


if __name__ == "__main__":
    sys.exit(main())
