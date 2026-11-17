"""Evidence-proportional weights, factor budgets, ramp cap, incremental alpha."""
from __future__ import annotations

import math

import numpy as np

from .posterior import ANN, p_positive, sr_se


def kelly_weights(posteriors: dict[str, tuple[float, float]], vols: dict[str, float],
                  min_p: float = 0.80, frac: float = 0.25) -> dict[str, float]:
    """w = frac * posterior-mean SR / sigma; 0 if P(SR>0) < min_p (or mean <= 0)."""
    out = {}
    for sid, (m, s) in posteriors.items():
        ok = p_positive(m, s) >= min_p and m > 0 and vols[sid] > 0
        out[sid] = frac * m / vols[sid] if ok else 0.0
    return out


def factor_budget(weights: dict[str, float], factor_of: dict[str, str],
                  budget: dict[str, float]) -> dict[str, float]:
    """Scale weights down within a factor so its total |weight| stays within budget."""
    totals: dict[str, float] = {}
    for sid, w in weights.items():
        f = factor_of[sid]
        totals[f] = totals.get(f, 0.0) + abs(w)
    out = {}
    for sid, w in weights.items():
        f = factor_of[sid]
        cap = budget.get(f, math.inf)
        out[sid] = w * min(1.0, cap / totals[f]) if totals[f] > 0 else w
    return out


def ramp_cap(base: float = 0.05, precision_now: float = 1.0,
             precision_at_launch: float = 1.0) -> float:
    """Capital-at-risk multiplier: base * sqrt(precision ratio), within [base, 1]."""
    ratio = max(precision_now / precision_at_launch, 1.0)
    return min(1.0, base * math.sqrt(ratio))


def incremental_alpha(signal_pnl, book_pnl) -> tuple[float, float]:
    """OLS of signal on book; returns (annualised residual-alpha SR, its se)."""
    y = np.asarray(signal_pnl, float)
    x = np.asarray(book_pnl, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sd = resid.std(ddof=2)
    if sd == 0:
        return 0.0, math.inf
    sr = float(beta[0] / sd * ANN)
    return sr, sr_se(sr, len(y) / 365)
