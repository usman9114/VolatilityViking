"""reports/YYYY-MM-DD.md: evidence table, targets vs held, orders, P&L, alerts."""
from __future__ import annotations

from pathlib import Path

from vv.runner.alerts import redact
from vv.runner.config import RunnerConfig, venue_symbol


def render(cfg: RunnerConfig, run_date: str, mode: str, res, st, active, variants, held, px,
           alerts: list[str]) -> str:
    L = [f"# Daily run {run_date} ({mode}, venue {cfg.venue})", ""]
    code = {0: "ok", 2: "DATA GAP, trading skipped"}.get(res.code, f"exit {res.code}")
    L += [f"Status: **{code}**", "", "## Evidence",
          "| signal | status | variant | post mean | post sd | P(SR>0) | fwd n | 1/4-Kelly | decay | final weight |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    status = {s.id: s.status for s in active}
    for e in res.evidence:
        v = variants[e.signal]
        var = f"{v.venue}{'/short' if v.allow_short else ''}{'/dial' if v.risk_dial else ''}"
        L.append(f"| {e.signal} | {status[e.signal]} | {var} | {e.post_mean:.3f} | {e.post_sd:.3f} | "
                 f"{e.p_pos:.3f} | {e.fwd_n} | {e.kelly_w:.3f} | {e.decay} | {e.weight:.4f} |")
    if not res.evidence:
        L.append("| (no signals with status in " + ", ".join(cfg.statuses) + ") | | | | | | | | | |")
    L += ["", "## Targets vs held",
          "| symbol | close | target weight | target qty | held qty |", "|---|---|---|---|---|"]
    for s in cfg.symbols:
        vs = venue_symbol(s, cfg.venue)
        w = res.book.get(s, 0.0)
        p = px.get(s)
        q = (w * cfg.dry_equity / p) if (p and mode == "dry") else float("nan")
        L.append(f"| {vs} | {p if p else '-'} | {w:.4f} | {q:.6f} | {held.get(vs, 0.0):.6f} |")
    L += ["", "## Orders"]
    if res.orders:
        L += ["| symbol | side | qty | price | status | client id |", "|---|---|---|---|---|---|"]
        L += [f"| {o.symbol} | {o.side} | {o.qty} | {o.price} | {o.status} | {o.client_id} |"
              for o in res.orders]
    else:
        L.append("No orders.")
    L += ["", "## Shadow P&L (daily, signal at weight 1)", "| signal | last date | pnl | days |", "|---|---|---|---|"]
    for s in active:
        series = st.pnl_series(s.id)
        if series:
            d, r = series[-1]
            L.append(f"| {s.id} | {d} | {r:+.5f} | {len(series)} |")
        else:
            L.append(f"| {s.id} | - | no realised days yet | 0 |")
    L += ["", "## Alerts"]
    L += [f"- {a}" for a in alerts] or ["None."]
    return "\n".join(L) + "\n"


def write_report(cfg, run_date, mode, res, st, active, variants, held, px, alerts) -> Path:
    out = Path(cfg.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{run_date}.md"
    p.write_text(redact(render(cfg, run_date, mode, res, st, active, variants, held, px, alerts)))
    return p
