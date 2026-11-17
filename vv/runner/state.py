"""SQLite (WAL) run state. The halt persists and never auto-resumes."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT, mode TEXT,
  started_ts REAL, status TEXT, exit_code INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS targets(bar_date TEXT, signal TEXT, symbol TEXT, weight REAL,
  PRIMARY KEY(bar_date, signal, symbol));
CREATE TABLE IF NOT EXISTS orders(client_id TEXT PRIMARY KEY, run_date TEXT, mode TEXT, symbol TEXT,
  side TEXT, qty REAL, price REAL, status TEXT, stage TEXT, reduce_only INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS fills(client_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, filled REAL,
  price REAL, ts REAL);
CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, qty REAL, updated_ts REAL);
CREATE TABLE IF NOT EXISTS shadow_pnl(bar_date TEXT, signal TEXT, pnl REAL, gross REAL, cost REAL,
  funding REAL, funding_missing INTEGER, PRIMARY KEY(bar_date, signal));
CREATE TABLE IF NOT EXISTS evidence(bar_date TEXT, signal TEXT, post_mean REAL, post_sd REAL, p_pos REAL,
  fwd_n INTEGER, kelly_w REAL, decay TEXT, weight REAL, PRIMARY KEY(bar_date, signal));
CREATE TABLE IF NOT EXISTS halt(id INTEGER PRIMARY KEY AUTOINCREMENT, set_ts REAL, reason TEXT,
  active INTEGER, cleared_ts REAL, clear_reason TEXT);
CREATE TABLE IF NOT EXISTS ingest(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, symbol TEXT,
  rows_added INTEGER, last_date TEXT, note TEXT);
"""


class State:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # halt ---------------------------------------------------------------
    def halt_active(self) -> bool:
        return self.db.execute("SELECT 1 FROM halt WHERE active=1 LIMIT 1").fetchone() is not None

    def halt_reason(self) -> str:
        r = self.db.execute("SELECT reason FROM halt WHERE active=1 ORDER BY id DESC LIMIT 1").fetchone()
        return r["reason"] if r else ""

    def set_halt(self, reason: str) -> None:
        with self.db:
            self.db.execute("INSERT INTO halt(set_ts, reason, active) VALUES(?,?,1)", (time.time(), reason))

    def resume(self, reason: str) -> int:
        """Clear every active halt (logged in the table). Returns how many were cleared."""
        with self.db:
            cur = self.db.execute(
                "UPDATE halt SET active=0, cleared_ts=?, clear_reason=? WHERE active=1", (time.time(), reason))
        return cur.rowcount

    # runs ---------------------------------------------------------------
    def start_run(self, run_date: str, mode: str) -> int:
        with self.db:
            cur = self.db.execute("INSERT INTO runs(run_date, mode, started_ts, status) VALUES(?,?,?,?)",
                                  (run_date, mode, time.time(), "running"))
        return int(cur.lastrowid)

    def end_run(self, run_id: int, status: str, code: int, note: str = "") -> None:
        with self.db:
            self.db.execute("UPDATE runs SET status=?, exit_code=?, note=? WHERE run_id=?",
                            (status, code, note, run_id))

    # targets / positions ------------------------------------------------
    def save_targets(self, bar_date: str, signal: str, weights: dict[str, float]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO targets(bar_date, signal, symbol, weight) VALUES(?,?,?,?)",
                [(bar_date, signal, s, float(w)) for s, w in sorted(weights.items())])

    def targets_on(self, bar_date: str, signal: str) -> dict[str, float]:
        rows = self.db.execute("SELECT symbol, weight FROM targets WHERE bar_date=? AND signal=?",
                               (bar_date, signal)).fetchall()
        return {r["symbol"]: r["weight"] for r in rows}

    def target_dates(self, signal: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT bar_date FROM targets WHERE signal=? ORDER BY bar_date", (signal,))]

    def positions(self) -> dict[str, float]:
        return {r["symbol"]: r["qty"] for r in self.db.execute("SELECT symbol, qty FROM positions")}

    def replace_positions(self, qty: dict[str, float]) -> None:
        with self.db:
            self.db.execute("DELETE FROM positions")
            self.db.executemany("INSERT INTO positions(symbol, qty, updated_ts) VALUES(?,?,?)",
                                [(s, float(q), time.time()) for s, q in sorted(qty.items()) if q])

    # orders / fills -----------------------------------------------------
    def save_order(self, run_date: str, mode: str, o) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (o.client_id, run_date, mode, o.symbol, o.side, o.qty, o.price, o.status, o.stage,
                 int(o.reduce_only), o.note))
            if o.filled:
                self.db.execute("INSERT OR REPLACE INTO fills VALUES(?,?,?,?,?,?)",
                                (o.client_id, o.symbol, o.side, o.filled, o.price, time.time()))

    # shadow pnl / evidence ----------------------------------------------
    def has_pnl(self, bar_date: str, signal: str) -> bool:
        return self.db.execute("SELECT 1 FROM shadow_pnl WHERE bar_date=? AND signal=?",
                               (bar_date, signal)).fetchone() is not None

    def save_pnl(self, bar_date: str, signal: str, pnl: float, gross: float, cost: float,
                 funding: float, funding_missing: bool) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO shadow_pnl VALUES(?,?,?,?,?,?,?)",
                            (bar_date, signal, pnl, gross, cost, funding, int(funding_missing)))

    def pnl_series(self, signal: str) -> list[tuple[str, float]]:
        return [(r["bar_date"], r["pnl"]) for r in self.db.execute(
            "SELECT bar_date, pnl FROM shadow_pnl WHERE signal=? ORDER BY bar_date", (signal,))]

    def save_evidence(self, bar_date: str, e) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO evidence VALUES(?,?,?,?,?,?,?,?,?)",
                            (bar_date, e.signal, e.post_mean, e.post_sd, e.p_pos, e.fwd_n, e.kelly_w,
                             e.decay, e.weight))

    def log_ingest(self, symbol: str, rows: int, last_date: str, note: str = "") -> None:
        with self.db:
            self.db.execute("INSERT INTO ingest(ts, symbol, rows_added, last_date, note) VALUES(?,?,?,?,?)",
                            (time.time(), symbol, rows, last_date, note))
