"""CUSUM decay monitor: realised minus prior-expected daily P&L."""
from __future__ import annotations


class CusumMonitor:
    """One-sided (downside) CUSUM on standardised P&L.

    State: ok -> halve (first breach) -> zero (second breach, terminal). The statistic
    resets after the first breach so the second needs fresh evidence.
    """

    def __init__(self, expected_daily_mean: float, sd: float, k: float = 0.5, h: float = 8.0):
        self.mu, self.sd, self.k, self.h = expected_daily_mean, sd, k, h
        self.s = 0.0
        self.breaches = 0

    @property
    def state(self) -> str:
        return ("ok", "halve", "zero")[min(self.breaches, 2)]

    def update(self, r: float) -> str:
        if self.breaches < 2:
            z = (r - self.mu) / self.sd
            self.s = max(0.0, self.s - z - self.k)
            if self.s > self.h:
                self.breaches += 1
                self.s = 0.0
        return self.state
