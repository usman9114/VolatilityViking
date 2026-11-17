"""Signals -> shadow P&L -> evidence weights -> combined book. Pure of venue I/O."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vv.core.trend import TrendParams, risk_dial, target_weights
from vv.evidence.decay import CusumMonitor
from vv.evidence.posterior import ANN, hierarchical_prior, p_positive, posterior
from vv.evidence.registry import Registry, Signal, ledger_srs, load_ledger
from vv.evidence.weights import factor_budget, kelly_weights, ramp_cap
from vv.runner.config import Caps, RunnerConfig
from vv.runner.state import State

FACTOR = "risk_on"  # every trend variant
DECAY_MULT = {"ok": 1.0, "halve": 0.5, "zero": 0.0}
FundingLoader = Callable[[str, pd.Timestamp], float | None]  # (symbol, bar_date) -> daily rate or None


@dataclass(frozen=True)
class Variant:
    venue: str
    allow_short: bool
    risk_dial: bool


@dataclass(frozen=True)
class EvidenceRow:
    signal: str
    post_mean: float
    post_sd: float
    p_pos: float
    fwd_n: int
    kelly_w: float
    decay: str
    weight: float


def variant_for(sig: Signal, root: str | Path) -> Variant:
    """Variant params come from the signal's pre-registration (venue, shorts, dial)."""
    p = Path(root) / sig.prereg_path
    v = json.loads(p.read_text()).get("variants", {}).get(sig.id) if p.is_file() else None
    if v is None:
        return Variant(sig.data, sig.data == "perp", False)
    return Variant(v["venue"], bool(v["allow_short"]), bool(v["risk_dial"]))


def active_signals(reg: Registry, statuses: list[str]) -> list[Signal]:
    return [s for s in reg.signals if s.status in statuses]


# ---------------------------------------------------------------- dial
def build_dial(close: pd.DataFrame, volume: pd.DataFrame, external_dir: str | Path,
               now: pd.Timestamp, max_age_days: int) -> tuple[pd.Series | None, str | None]:
    """Risk dial from the feature screen's builders. (None, reason) => no penalty, caller alerts."""
    from research import feature_screen as fs  # heavy import (scipy); only when a dial variant is active

    ext = Path(external_dir)
    exp = now.tz_convert("UTC").normalize() - pd.Timedelta(days=1)
    # metrics exist for a subset of the basket (BTC/ETH/SOL); the builder averages what is present
    metrics = [ext / f"binance_metrics_{s}.parquet" for s in fs.SYMS]
    need = [ext / "fred.parquet", ext / "stablecoins.parquet", metrics[0]]
    need += [m for m in metrics[1:] if m.exists()]
    stale = []
    frames: dict[str, pd.DataFrame] = {}
    for p in need:
        if not p.exists():
            stale.append(f"{p.name} missing")
            continue
        df = fs._utc(pd.read_parquet(p))
        frames[p.name] = df
        if (exp - df.index.max()).days > max_age_days:
            stale.append(f"{p.name} ends {df.index.max().date()}")
    if stale:
        return None, "dial inputs stale/missing, no penalty applied: " + "; ".join(stale)
    if any(s not in close.columns for s in fs.SYMS):
        return None, "dial needs the full feature-screen basket in `symbols`, no penalty applied"
    try:
        def rd(name):
            p = ext / name
            return fs._utc(pd.read_parquet(p)) if p.exists() else None

        inp = {"prices": {s: pd.DataFrame({"close": close[s], "volume": volume[s]}).dropna()
                          for s in fs.SYMS},
               "fred": frames["fred.parquet"], "fear_greed": rd("fear_greed.parquet"),
               "stablecoins": frames["stablecoins.parquet"], "metrics": {}, "premium": {}, "funding": {}}
        for s in fs.SYMS:
            for key, name in (("metrics", f"binance_metrics_{s}.parquet"),
                              ("premium", f"binance_premium_{s}.parquet"), ("funding", f"funding_{s}.parquet")):
                df = rd(name)
                if df is not None:
                    inp[key][s] = df
        feats = fs.build_features(inp, close.index)["basket"]
        return risk_dial(feats), None
    except Exception as e:
        return None, f"dial build failed ({type(e).__name__}), no penalty applied"


# ---------------------------------------------------------------- signal targets
def signal_targets(close: pd.DataFrame, var: Variant, caps: Caps,
                   dial: pd.Series | None) -> pd.Series:
    """Target weights at the last closed bar for one signal."""
    params = TrendParams(per_coin_cap=caps.per_coin, gross_cap=caps.gross, allow_short=var.allow_short)
    w = target_weights(close, params, dial if var.risk_dial else None)
    return w.iloc[-1].fillna(0.0)


# ---------------------------------------------------------------- shadow P&L
def update_shadow(state: State, signal: str, var: Variant, close: pd.DataFrame, funding: FundingLoader,
                  cost_bps: float) -> list[str]:
    """Realise P&L for each consecutive pair of stored target dates (t-1 targets, t closes).

    Returns notes about missing funding (estimated 0 and flagged)."""
    notes: list[str] = []
    dates = state.target_dates(signal)
    for prev, cur in zip(dates, dates[1:], strict=False):
        pd_, cd = pd.Timestamp(prev, tz="UTC"), pd.Timestamp(cur, tz="UTC")
        if cd - pd_ != pd.Timedelta(days=1) or state.has_pnl(cur, signal):
            continue
        w = state.targets_on(prev, signal)
        w_prev2 = state.targets_on((pd_ - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), signal)
        if pd_ not in close.index or cd not in close.index:
            continue
        ret = close.loc[cd] / close.loc[pd_] - 1.0
        gross = sum(wt * float(ret.get(s, 0.0)) for s, wt in w.items() if np.isfinite(ret.get(s, np.nan)))
        turnover = sum(abs(w.get(s, 0.0) - w_prev2.get(s, 0.0)) for s in set(w) | set(w_prev2))
        cost = turnover * cost_bps / 1e4
        fund, missing = 0.0, False
        if var.venue == "perp":
            for s, wt in w.items():
                if not wt:
                    continue
                rate = funding(s, cd)
                if rate is None:
                    missing = True
                else:
                    fund -= wt * rate  # longs pay positive funding
            if missing:
                notes.append(f"{signal} {cur}: funding unavailable, estimated 0")
        state.save_pnl(cur, signal, gross - cost + fund, gross, cost, fund, missing)
    return notes


def funding_from_archive(external_dir: str | Path) -> FundingLoader:
    cache: dict[str, pd.DataFrame | None] = {}

    def load(symbol: str, bar_date: pd.Timestamp) -> float | None:
        if symbol not in cache:
            p = Path(external_dir) / f"funding_{symbol}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index, utc=True).normalize()
                cache[symbol] = df
            else:
                cache[symbol] = None
        df = cache[symbol]
        if df is None or bar_date not in df.index:
            return None
        return float(df.loc[bar_date, "funding_rate"])

    return load


def funding_with_live(external_dir: str | Path,
                      fetch: Callable[[str, int], pd.Series] | None = None,
                      lookback_days: int = 60) -> FundingLoader:
    """Archive first; if the archive lacks the day, fall back to the live USD-M funding API
    (reachable from the operator's VM). Daily rate = sum of that UTC day's funding prints."""
    archive = funding_from_archive(external_dir)
    if fetch is None:
        from vv.runner.data import fetch_funding as fetch
    live: dict[str, pd.Series | None] = {}

    def load(symbol: str, bar_date: pd.Timestamp) -> float | None:
        v = archive(symbol, bar_date)
        if v is not None:
            return v
        if symbol not in live:
            since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
            s = fetch(symbol, int(since.timestamp() * 1000))
            live[symbol] = s.groupby(s.index.normalize()).sum() if len(s) else None
        s = live[symbol]
        if s is None or bar_date not in s.index:
            return None
        return float(s.loc[bar_date])

    return load


# ---------------------------------------------------------------- evidence
def compute_evidence(reg_signals: list[Signal], state: State, cfg: RunnerConfig,
                     ledger_path: str | Path, *, default_vol: float = 0.12) -> list[EvidenceRow]:
    """Posterior + CUSUM per signal, then 1/4-Kelly, factor budget and ramp cap -> final weight."""
    p = Path(ledger_path)
    prior = hierarchical_prior(*ledger_srs(load_ledger(p))) if p.exists() else (0.0, 0.5)
    posts: dict[str, tuple[float, float]] = {}
    vols: dict[str, float] = {}
    base: dict[str, tuple[float, float]] = {}
    fwds: dict[str, list[float]] = {}
    decay: dict[str, str] = {}
    for s in reg_signals:
        fwd = [r for _, r in state.pnl_series(s.id)]
        fwds[s.id] = fwd
        t1 = (s.t1.sr_oos_stress, s.t1.se) if s.t1 else None
        base[s.id] = posterior(prior, t1, None)
        posts[s.id] = posterior(prior, t1, fwd)
        vols[s.id] = float(np.std(fwd, ddof=1) * ANN) if len(fwd) >= 30 and np.std(fwd) > 0 else default_vol
        sd_d = vols[s.id] / ANN
        mon = CusumMonitor(base[s.id][0] / ANN * sd_d, sd_d)
        for r in fwd:
            mon.update(r)
        decay[s.id] = mon.state
    kelly = kelly_weights(posts, vols)
    budgeted = factor_budget(kelly, {s.id: FACTOR for s in reg_signals}, cfg.factor_budget)
    rows = []
    for s in reg_signals:
        m, sd = posts[s.id]
        ramp = ramp_cap(cfg.ramp_base, 1 / sd**2, 1 / base[s.id][1] ** 2)
        w = budgeted[s.id] * ramp * DECAY_MULT[decay[s.id]]
        rows.append(EvidenceRow(s.id, m, sd, p_positive(m, sd), len(fwds[s.id]), kelly[s.id],
                                decay[s.id], w))
    return rows


# ---------------------------------------------------------------- book
def apply_caps(w: pd.Series, caps: Caps, allow_short: bool) -> pd.Series:
    """Per-coin cap, shorts <= caps.short, gross <= caps.gross (only ever scales down)."""
    w = w.clip(-caps.per_coin, caps.per_coin).astype(float)
    if not allow_short:
        w = w.clip(lower=0.0)
    short = float(-w[w < 0].sum())
    if short > caps.short > 0 or (short > 0 and caps.short <= 0):
        f = caps.short / short if caps.short > 0 else 0.0
        w = w.where(w >= 0, w * f)
    gross = float(w.abs().sum())
    if gross > caps.gross:
        w = w * (caps.gross / gross)
    return w


def combine_book(targets: dict[str, pd.Series], weights: dict[str, float], symbols: list[str],
                 caps: Caps, allow_short: bool) -> pd.Series:
    book = pd.Series(0.0, index=symbols)
    for sid, t in targets.items():
        book = book.add(t.reindex(symbols).fillna(0.0) * weights.get(sid, 0.0), fill_value=0.0)
    out = apply_caps(book, caps, allow_short)
    return out.where(out.abs() > 1e-12, 0.0)


