from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from vv.core.params import GridParams, RiskParams
from vv.core.types import (
    CancelOrder,
    HaltState,
    OpenOrder,
    PlaceOrder,
    PortfolioState,
    Position,
    Snapshot,
    SymbolFilters,
    SymbolMarket,
)

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
FILT = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)


def _order(symbol, side, price, qty, filled=0.0, cid="vv-x"):
    return OpenOrder(
        symbol=symbol, side=side, price=price, qty=qty, filled=filled,
        client_id=cid, exchange_id="1", ts=TS,
    )


def _portfolio(**kw):
    base = dict(
        ts=TS,
        cash_free=700.0,
        cash_reserved=100.0,
        positions={
            "BTC/USDT": Position("BTC/USDT", qty=0.002, avg_entry=50_000.0, realized_pnl=0.0),
            "SOL/USDT": Position("SOL/USDT", qty=2.0, avg_entry=100.0, realized_pnl=1.5),
        },
        open_orders=(
            _order("BTC/USDT", "buy", 49_000.0, 0.001, cid="vv-a"),  # 49 notional
            _order("SOL/USDT", "buy", 90.0, 1.0, filled=0.5, cid="vv-b"),  # 45 remaining
            _order("SOL/USDT", "sell", 120.0, 1.0, cid="vv-c"),  # sells excluded
        ),
        peak_equity=1000.0,
        halt=HaltState(active=False, reason=None, since=None),
    )
    base.update(kw)
    return PortfolioState(**base)


MARKS = {"BTC/USDT": 51_000.0, "SOL/USDT": 110.0}


def test_equity_two_symbols_includes_reserved_cash():
    pf = _portfolio()
    # 700 free + 100 reserved + 0.002*51000 (102) + 2*110 (220)
    assert pf.equity(MARKS) == pytest.approx(1122.0)


def test_exposure_holdings_plus_resting_buy_notional():
    pf = _portfolio()
    holdings = 102.0 + 220.0
    resting_buys = 49.0 + 0.5 * 90.0
    assert pf.exposure(MARKS) == pytest.approx(holdings + resting_buys)


def test_exposure_by_symbol():
    pf = _portfolio()
    by = pf.exposure_by_symbol(MARKS)
    assert by["BTC/USDT"] == pytest.approx(102.0 + 49.0)
    assert by["SOL/USDT"] == pytest.approx(220.0 + 45.0)
    assert sum(by.values()) == pytest.approx(pf.exposure(MARKS))


def test_exposure_by_symbol_includes_symbol_with_only_resting_buy():
    pf = _portfolio(
        positions={},
        open_orders=(_order("BTC/USDT", "buy", 50_000.0, 0.01),),
    )
    assert pf.exposure_by_symbol(MARKS) == {"BTC/USDT": pytest.approx(500.0)}


def test_empty_portfolio():
    pf = _portfolio(cash_free=1000.0, cash_reserved=0.0, positions={}, open_orders=())
    assert pf.equity({}) == 1000.0
    assert pf.exposure({}) == 0.0
    assert pf.exposure_by_symbol({}) == {}


def test_missing_mark_for_held_position_raises():
    with pytest.raises(KeyError):
        _portfolio().equity({"BTC/USDT": 51_000.0})


def test_types_are_frozen():
    pf = _portfolio()
    with pytest.raises(FrozenInstanceError):
        pf.cash_free = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        PlaceOrder("BTC/USDT", "buy", 1.0, 1.0, "vv-1").qty = 2.0  # type: ignore[misc]


def test_place_order_defaults_post_only():
    assert PlaceOrder("BTC/USDT", "buy", 1.0, 1.0, "vv-1").post_only is True
    assert CancelOrder("BTC/USDT", "vv-1").client_id == "vv-1"


def test_snapshot_fresh_iff_no_stale():
    mkt = SymbolMarket("BTC/USDT", TS, bid=50_000.0, ask=50_001.0, last=50_000.5, filters=FILT)
    bars = {"BTC/USDT": pd.DataFrame({"close": [1.0]})}
    ok = Snapshot(TS, {"BTC/USDT": mkt}, bars, _portfolio(), stale=())
    bad = Snapshot(TS, {"BTC/USDT": mkt}, bars, _portfolio(), stale=("SOL/USDT",))
    assert ok.fresh is True
    assert bad.fresh is False


# --- params ---------------------------------------------------------------

RISK = dict(
    total_exposure_cap=0.4,
    per_symbol_cap=0.15,
    portfolio_halt_dd=0.15,
    symbol_halt_dd=0.25,
)
GRID = dict(spacing_sigmas=1.0, levels=4, qty_multiplier=1.2, maker_fee=0.001)


def test_risk_params_valid():
    r = RiskParams(**RISK)
    assert r.total_exposure_cap == 0.4
    with pytest.raises(ValidationError):
        r.total_exposure_cap = 0.5  # frozen


def test_total_cap_above_0_6_rejected():
    with pytest.raises(ValidationError):
        RiskParams(**{**RISK, "total_exposure_cap": 0.61})
    assert RiskParams(**{**RISK, "total_exposure_cap": 0.6}).total_exposure_cap == 0.6


@pytest.mark.parametrize("missing", list(RISK))
def test_risk_fields_have_no_defaults(missing):
    kw = {k: v for k, v in RISK.items() if k != missing}
    with pytest.raises(ValidationError):
        RiskParams(**kw)


def test_risk_params_range_validation():
    for bad in (
        {"total_exposure_cap": 0.0},
        {"per_symbol_cap": 0.0},
        {"per_symbol_cap": 0.5},  # > total cap
        {"portfolio_halt_dd": 0.0},
        {"portfolio_halt_dd": 1.0},
        {"symbol_halt_dd": 1.5},
    ):
        with pytest.raises(ValidationError):
            RiskParams(**{**RISK, **bad})


def test_risk_params_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        RiskParams(**RISK, surprise=1)


def test_grid_params_valid_and_defaults():
    g = GridParams(**GRID, vol_scalar={"BTC/USDT": 1.0, "DOGE/USDT": 1.4})
    assert g.tp_fee_multiple == 2.5
    assert g.vol_scalar["DOGE/USDT"] == 1.4
    with pytest.raises(ValidationError):
        g.levels = 5


def test_grid_params_range_validation():
    for bad in (
        {"spacing_sigmas": 0.0},
        {"levels": 0},
        {"qty_multiplier": 0.0},
        {"maker_fee": -0.001},
        {"maker_fee": 0.05},
        {"tp_fee_multiple": 0.0},
        {"vol_scalar": {"BTC/USDT": 0.0}},
    ):
        with pytest.raises(ValidationError):
            GridParams(**{**GRID, **bad})
