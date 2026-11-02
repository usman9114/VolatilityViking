"""Strategy and risk parameters (pydantic v2, frozen).

Grid spacing is in sigma units (per-bar volatility estimate) and shared by
all coins; the only per-coin difference is ``vol_scalar``.
Risk fields have deliberately **no defaults**: callers must choose them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_TOTAL_EXPOSURE_CAP = 0.6


class GridParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spacing_sigmas: float = Field(gt=0, le=20)
    min_step: float = Field(ge=0.001, le=0.2)  # fraction; floor on grid step
    sigma_horizon_minutes: int = Field(ge=5, le=1440)  # horizon over which sigma_h is measured
    levels: int = Field(ge=1, le=50)
    qty_multiplier: float = Field(gt=0, le=10)
    tp_fee_multiple: float = Field(default=2.5, gt=0, le=100)
    maker_fee: float = Field(ge=0, le=0.01)  # fraction per fill, e.g. 0.001
    vol_scalar: dict[str, float] = Field(default_factory=dict)  # symbol -> scalar, absent == 1.0

    @property
    def round_trip_fee(self) -> float:
        return 2 * self.maker_fee

    @field_validator("vol_scalar")
    @classmethod
    def _scalars_positive(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: x for k, x in v.items() if not 0 < x <= 100}
        if bad:
            raise ValueError(f"vol_scalar values must be in (0, 100]: {bad}")
        return v


class RiskParams(BaseModel):
    """All values are fractions of equity (drawdowns: fraction below peak)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_exposure_cap: float = Field(gt=0, le=MAX_TOTAL_EXPOSURE_CAP)
    per_symbol_cap: float = Field(gt=0, le=MAX_TOTAL_EXPOSURE_CAP)
    portfolio_halt_dd: float = Field(gt=0, lt=1)
    symbol_halt_dd: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _per_symbol_within_total(self) -> RiskParams:
        if self.per_symbol_cap > self.total_exposure_cap:
            raise ValueError("per_symbol_cap must be <= total_exposure_cap")
        return self
