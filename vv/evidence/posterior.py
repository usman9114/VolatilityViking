"""Hierarchical prior and normal-normal posterior for annualised Sharpe ratios."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ANN = math.sqrt(365)
DEFAULT_PRIOR = (0.0, 0.5)
MIN_LEDGER = 5
TAU_FLOOR = 0.05


def hierarchical_prior(ledger_srs: Sequence[float], ledger_ses: Sequence[float]) -> tuple[float, float]:
    """Method-of-moments empirical Bayes: mu = mean SR, tau^2 = var(SR) - mean(se^2)."""
    n = len(ledger_srs)
    if n < MIN_LEDGER or n != len(ledger_ses):
        return DEFAULT_PRIOR
    srs = np.asarray(ledger_srs, float)
    ses = np.asarray(ledger_ses, float)
    tau2 = float(srs.var(ddof=1) - np.mean(ses**2))
    return float(srs.mean()), max(math.sqrt(max(tau2, 0.0)), TAU_FLOOR)


def sr_se(sr: float, years: float) -> float:
    return math.sqrt((1 + 0.5 * sr**2) / years)


def _update(mean: float, sd: float, obs: float, se: float) -> tuple[float, float]:
    p0, p1 = 1 / sd**2, 1 / se**2
    return (mean * p0 + obs * p1) / (p0 + p1), math.sqrt(1 / (p0 + p1))


def posterior(prior: tuple[float, float], t1: tuple[float, float] | None,
              forward_daily_returns: Sequence[float] | None,
              eff_n_factor: float = 1.0) -> tuple[float, float]:
    """Normal-normal update of the prior with T1 (sr, se) then forward daily returns.

    eff_n_factor in (0, 1] scales the effective sample size for overlapping horizons.
    """
    mean, sd = prior
    if t1 is not None:
        mean, sd = _update(mean, sd, t1[0], t1[1])
    if forward_daily_returns is not None and len(forward_daily_returns) >= 2:
        r = np.asarray(forward_daily_returns, float)
        std = r.std(ddof=1)
        if std > 0:
            sr = float(r.mean() / std * ANN)
            years = len(r) * eff_n_factor / 365
            mean, sd = _update(mean, sd, sr, sr_se(sr, years))
    return mean, sd


def p_positive(mean: float, sd: float) -> float:
    return 0.5 * math.erfc(-mean / sd / math.sqrt(2))
