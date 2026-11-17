"""Signal registry (YAML, pydantic-validated) and append-only trial ledger (CSV)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

LEDGER_FIELDS = ["id", "date", "family", "outcome", "sr", "se"]


class PreregMismatch(Exception):
    """A pre-registration file no longer matches its recorded sha256."""


class T1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sr_oos_stress: float
    se: float
    years: float


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    family: str
    role: Literal["directional", "risk_off"]
    hypothesis: str
    sign: int
    horizon: str
    data: str
    prereg_path: str
    prereg_sha256: str
    status: Literal["candidate", "shadow", "live", "retired"]
    created: str
    t1: T1 | None = None
    notes: str = ""


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signals: list[Signal]

    def get(self, sid: str) -> Signal:
        return next(s for s in self.signals if s.id == sid)


def load_registry(path: str | Path) -> Registry:
    reg = Registry.model_validate(yaml.safe_load(Path(path).read_text()))
    ids = [s.id for s in reg.signals]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate signal ids in registry")
    return reg


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_prereg(registry: Registry, root: str | Path = ".") -> None:
    """Raise PreregMismatch if any prereg file is missing or its hash differs."""
    for s in registry.signals:
        p = Path(root) / s.prereg_path
        if not p.is_file():
            raise PreregMismatch(f"{s.id}: prereg file missing: {s.prereg_path}")
        if sha256_file(p) != s.prereg_sha256:
            raise PreregMismatch(f"{s.id}: sha256 mismatch for {s.prereg_path}")


def load_ledger(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_ledger(path: str | Path, *, id: str, date: str, family: str, outcome: str,
                  sr: float | None = None, se: float | None = None) -> None:
    """Append one row. Existing rows are never rewritten; duplicate ids are refused."""
    if any(r["id"] == id for r in load_ledger(path)):
        raise ValueError(f"ledger already has id {id!r}")
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, LEDGER_FIELDS).writerow(
            {"id": id, "date": date, "family": family, "outcome": outcome,
             "sr": "" if sr is None else sr, "se": "" if se is None else se})


def ledger_srs(rows: list[dict]) -> tuple[list[float], list[float]]:
    """(srs, ses) for ledger rows that carry a comparable SR estimate."""
    pairs = [(float(r["sr"]), float(r["se"])) for r in rows if r.get("sr") and r.get("se")]
    return [p[0] for p in pairs], [p[1] for p in pairs]
