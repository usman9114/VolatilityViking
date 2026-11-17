"""One daily run: data -> signals -> evidence -> book -> orders -> state/report."""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vv.evidence.registry import PreregMismatch, load_registry, verify_prereg
from vv.exec.common import OrderResult, StartupCheckFailed, TradingHalted, client_order_id
from vv.exec.reconcile import reconcile
from vv.runner import data as D
from vv.runner import engine as E
from vv.runner.alerts import Alerter, install_log_redaction
from vv.runner.config import RunnerConfig, venue_symbol
from vv.runner.report import write_report
from vv.runner.state import State

LOG = logging.getLogger("vv.runner")
OK, ERROR, GAP, HALTED, RECONCILE_HALT, LIVE_REFUSED = 0, 1, 2, 3, 4, 5


@dataclass
class RunResult:
    code: int
    orders: list[OrderResult] = field(default_factory=list)
    book: dict[str, float] = field(default_factory=dict)
    evidence: list[E.EvidenceRow] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


def default_adapter(cfg: RunnerConfig, mode: str, halt_active: Callable[[], bool]):
    if cfg.venue == "spot":
        if mode != "live":
            raise RuntimeError("spot testnet is not supported by vv.exec; use venue: perp or dry mode")
        from vv.exec.binance_spot import SpotAdapter
        return SpotAdapter(halt_active=halt_active)
    from vv.exec.binance_perp import PerpAdapter
    return PerpAdapter(mode=mode, halt_active=halt_active)


def _dry_orders(cfg: RunnerConfig, qty: dict[str, float], held: dict[str, float],
                price: dict[str, float], decision_ts: int) -> list[OrderResult]:
    out = []
    for vs in sorted(qty):
        delta = round(qty[vs] - held.get(vs, 0.0), 6)
        if abs(delta) * price[vs] < cfg.min_order_notional:
            continue
        side = "buy" if delta > 0 else "sell"
        cid = client_order_id(vs, side, abs(delta), decision_ts, "maker")
        out.append(OrderResult(vs, side, abs(delta), price[vs], cid, "dry_run"))
    return out


def run(cfg: RunnerConfig, mode: str | None = None, *, now: pd.Timestamp | None = None,
        fetch: D.Fetcher = D.fetch_daily, adapter_factory=default_adapter,
        alerter: Alerter | None = None, env: Mapping[str, str] | None = None) -> RunResult:
    env = os.environ if env is None else env
    mode = mode or cfg.mode
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    alerter = alerter or Alerter(env)
    install_log_redaction(env)
    res = RunResult(OK)

    def finish(code: int, st: State | None = None, run_id: int | None = None, status: str = "",
               note: str = "") -> RunResult:
        res.code, res.alerts = code, list(alerter.messages)
        if st is not None and run_id is not None:
            st.end_run(run_id, status or ("ok" if code == OK else "failed"), code, note)
        alerter.heartbeat(code == OK)
        return res

    # live guard comes before anything else, including opening state
    if mode == "live" and env.get("VV_LIVE_CONFIRM") != "yes":
        alerter.alert("live mode refused: VV_LIVE_CONFIRM=yes is not set")
        return finish(LIVE_REFUSED)

    st = State(cfg.state_db)
    run_date = now.strftime("%Y-%m-%d")
    run_id = st.start_run(run_date, mode)
    try:
        if st.halt_active():
            alerter.alert(f"HALTED, not trading: {st.halt_reason()} (clear with `resume --reason ...`)")
            return finish(HALTED, st, run_id, "halted")

        reg = load_registry(Path(cfg.repo_root) / cfg.registry_path)
        try:
            verify_prereg(reg, cfg.repo_root)
        except PreregMismatch as e:
            alerter.alert(f"prereg check failed, not trading: {e}")
            return finish(ERROR, st, run_id, "prereg_mismatch")

        # 1. data -------------------------------------------------------
        store = D.DailyStore(cfg.data_dir)
        added, errs = D.update_store(store, cfg.symbols, fetch, now, cfg.history_days)
        bar = D.expected_last_bar(now)
        for s, n in added.items():
            st.log_ingest(s, n, str(bar.date()))
        close, volume = D.wide(store, cfg.symbols, now)
        gaps = D.detect_gaps(close, now)
        if gaps:
            alerter.alert("DATA GAP, trading skipped: " + "; ".join(errs + gaps))
            write_report(cfg, run_date, mode, res, st, [], {}, {}, {}, alerter.messages)
            return finish(GAP, st, run_id, "gap")
        bar_s = str(bar.date())

        # 2. signals ----------------------------------------------------
        active = E.active_signals(reg, cfg.statuses)
        variants = {s.id: E.variant_for(s, cfg.repo_root) for s in active}
        dial = None
        if any(v.risk_dial for v in variants.values()):
            dial, why = E.build_dial(close, volume, cfg.external_dir, now, cfg.external_max_age_days)
            if why:
                alerter.alert(why)
        targets = {s.id: E.signal_targets(close, variants[s.id], cfg.caps, dial) for s in active}
        for sid, t in targets.items():
            st.save_targets(bar_s, sid, {k: float(v) for k, v in t.items()})

        # 3. evidence ---------------------------------------------------
        funding = E.funding_from_archive(cfg.external_dir)
        notes = []
        for s in active:
            notes += E.update_shadow(st, s.id, variants[s.id], close, funding, cfg.shadow_cost_bps)
        if notes:
            alerter.alert("funding unavailable for %d signal-day(s); estimated 0 (first: %s)"
                          % (len(notes), notes[0]))
        res.evidence = E.compute_evidence(active, st, cfg, Path(cfg.repo_root) / cfg.ledger_path)
        for e in res.evidence:
            st.save_evidence(bar_s, e)
        weights = {e.signal: e.weight for e in res.evidence}

        # 4. book (only `live` signals of the configured venue carry capital) ----
        allow_short = cfg.venue == "perp"
        tradable = {s.id: targets[s.id] for s in active
                    if s.status == "live" and variants[s.id].venue == cfg.venue}
        book = E.combine_book(tradable, weights, cfg.symbols, cfg.caps, allow_short)
        st.save_targets(bar_s, "BOOK", {k: float(v) for k, v in book.items()})
        res.book = {k: float(v) for k, v in book.items()}
        px = {s: float(close[s].iloc[-1]) for s in cfg.symbols}
        decision_ts = int((bar + pd.Timedelta(days=1)).timestamp())
        vsym = {s: venue_symbol(s, cfg.venue) for s in cfg.symbols}
        vprice = {vsym[s]: px[s] for s in cfg.symbols}

        # 5. execute ----------------------------------------------------
        held = st.positions()
        if mode == "dry":
            qty = {vsym[s]: book[s] * cfg.dry_equity / px[s] for s in cfg.symbols}
            res.orders = _dry_orders(cfg, qty, held, vprice, decision_ts)
        else:
            try:
                ad = adapter_factory(cfg, mode, st.halt_active)
                ad.start(list(vsym.values()))
                diffs = reconcile(ad.positions(), held, tol=1e-8)
            except StartupCheckFailed as e:
                st.set_halt(f"startup check failed: {e}")
                alerter.alert(f"HALT: startup check failed: {e}")
                return finish(RECONCILE_HALT, st, run_id, "startup_failed")
            if diffs:
                msg = "; ".join(f"{d.symbol} {d.kind} expected {d.expected} actual {d.actual}" for d in diffs)
                st.set_halt(f"reconcile diff: {msg}")
                alerter.alert(f"HALT: reconcile diff, not trading: {msg}")
                return finish(RECONCILE_HALT, st, run_id, "reconcile_halt")
            equity = ad.balance().margin
            qty = {vsym[s]: book[s] * equity / px[s] for s in cfg.symbols}
            try:
                res.orders = ad.rebalance(
                    qty, max_short_frac=cfg.caps.short, gross_cap=cfg.caps.gross, decision_ts=decision_ts)
            except TradingHalted as e:
                alerter.alert(f"HALTED mid-run: {e}")
                return finish(HALTED, st, run_id, "halted")
            except Exception as e:
                st.set_halt(f"execution error ({type(e).__name__}); state unknown")
                alerter.alert(f"HALT: execution error {type(e).__name__}; orders may be live, reconcile manually")
                return finish(ERROR, st, run_id, "exec_error")
            st.replace_positions({p.symbol: p.qty for p in ad.positions().values()})
        for o in res.orders:
            st.save_order(run_date, mode, o)

        # 6/8. report ---------------------------------------------------
        write_report(cfg, run_date, mode, res, st, active, variants, held, px, alerter.messages)
        return finish(OK, st, run_id, "ok")
    except Exception as e:
        LOG.exception("run failed")
        alerter.alert(f"run failed: {type(e).__name__}: {e}")
        return finish(ERROR, st, run_id, "error", type(e).__name__)
    finally:
        st.close()


