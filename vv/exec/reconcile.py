"""Compare exchange positions with the state store's expected positions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vv.exec.common import PositionInfo


@dataclass(frozen=True)
class Diff:
    symbol: str
    expected: float
    actual: float
    kind: str  # missing | unexpected | size

    @property
    def delta(self) -> float:
        return self.actual - self.expected


def reconcile(
    actual: Mapping[str, PositionInfo | float],
    expected: Mapping[str, float],
    *,
    tol: Mapping[str, float] | float = 1e-9,
    inflight: Mapping[str, float] | None = None,
) -> list[Diff]:
    """Return unexplained diffs. ``inflight`` is the signed qty of orders known to be
    working (expected + inflight is what the exchange may legitimately show).
    Any non-empty result means halt + alert; the caller decides.
    """
    act = {s: (p.qty if isinstance(p, PositionInfo) else float(p)) for s, p in actual.items()}
    inf = inflight or {}
    diffs: list[Diff] = []
    for s in sorted(set(act) | set(expected)):
        a, e = act.get(s, 0.0), expected.get(s, 0.0)
        t = tol.get(s, 1e-9) if isinstance(tol, Mapping) else tol
        ok = abs(a - e) <= t or abs(a - (e + inf.get(s, 0.0))) <= t
        if ok:
            continue
        kind = "unexpected" if abs(e) <= t else "missing" if abs(a) <= t else "size"
        diffs.append(Diff(s, e, a, kind))
    return diffs
