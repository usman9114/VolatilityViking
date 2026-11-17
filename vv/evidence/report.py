"""Markdown evidence report. Run: python -m vv.evidence.report"""
from __future__ import annotations

import math
from pathlib import Path

from .posterior import hierarchical_prior, p_positive, posterior
from .registry import Registry, ledger_srs, load_ledger, load_registry
from .weights import kelly_weights


def build_report(registry: Registry, prior: tuple[float, float],
                 forward: dict[str, list[float]] | None = None,
                 vols: dict[str, float] | None = None,
                 weights: dict[str, float] | None = None,
                 decay: dict[str, str] | None = None,
                 eff_n_factor: float = 1.0) -> str:
    forward = forward or {}
    decay = decay or {}
    posts, fwd_n = {}, {}
    for s in registry.signals:
        t1 = (s.t1.sr_oos_stress, s.t1.se) if s.t1 else None
        f = forward.get(s.id)
        fwd_n[s.id] = len(f) if f is not None else 0
        posts[s.id] = posterior(prior, t1, f, eff_n_factor)
    if weights is None:
        vols = vols or {s.id: 0.10 for s in registry.signals}  # assumed sigma, annualised
        weights = kelly_weights(posts, vols)
    lines = [f"Prior N({prior[0]:.2f}, {prior[1]:.2f}^2)", "",
             "| signal | prior | T1 SR (se) | fwd n | post mean | post sd | P(SR>0) | weight | decay |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in registry.signals:
        m, sd = posts[s.id]
        t1 = f"{s.t1.sr_oos_stress:.2f} ({s.t1.se:.2f})" if s.t1 else "-"
        lines.append(f"| {s.id} | {prior[0]:.2f}/{prior[1]:.2f} | {t1} | {fwd_n[s.id]} | {m:.3f} | "
                     f"{sd:.3f} | {p_positive(m, sd):.3f} | {weights[s.id]:.3f} | "
                     f"{decay.get(s.id, 'ok')} |")
    return "\n".join(lines)


def main(root: str = ".") -> None:
    root_p = Path(root)
    reg = load_registry(root_p / "research/signals/registry.yaml")
    srs, ses = ledger_srs(load_ledger(root_p / "research/signals/ledger.csv"))
    print(build_report(reg, hierarchical_prior(srs, ses)))


if __name__ == "__main__":
    main()
