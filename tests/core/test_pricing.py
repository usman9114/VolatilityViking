import pytest

from vv.core.params import GridParams
from vv.core.pricing import buy_ladder, step_size, take_profit_price
from vv.core.types import SymbolFeatures


def _params(**kw) -> GridParams:
    base = dict(
        spacing_sigmas=1.5, min_step=0.004, sigma_horizon_minutes=60,
        levels=5, qty_multiplier=1.2, maker_fee=0.001, tp_fee_multiple=2.5,
    )
    return GridParams(**{**base, **kw})


def _feat(sigma_h: float = 0.01, close: float = 100.0) -> SymbolFeatures:
    return SymbolFeatures(close=close, sigma_h=sigma_h, ema_4h=close, regime_ok=True, warm=True)


def test_round_trip_fee_property():
    assert _params(maker_fee=0.00075).round_trip_fee == pytest.approx(0.0015)


def test_params_new_field_ranges():
    from pydantic import ValidationError

    for bad in ({"min_step": 0.0005}, {"min_step": 0.3}, {"sigma_horizon_minutes": 4},
                {"sigma_horizon_minutes": 1441}):
        with pytest.raises(ValidationError):
            _params(**bad)


def test_step_size_sigma_floor_dominates():
    p = _params()  # min_step .004, fee floor 2.5*.002=.005
    assert step_size(_feat(0.01), p, "BTC/USDT") == pytest.approx(0.015)


def test_step_size_min_step_floor():
    p = _params(min_step=0.02)
    assert step_size(_feat(0.001), p, "BTC/USDT") == pytest.approx(0.02)


def test_step_size_fee_floor():
    p = _params(min_step=0.001, tp_fee_multiple=5.0)  # 5 * .002 = .01
    assert step_size(_feat(0.001), p, "BTC/USDT") == pytest.approx(0.01)


def test_step_size_vol_scalar():
    p = _params(vol_scalar={"DOGE/USDT": 2.0})
    f = _feat(0.01)
    assert step_size(f, p, "DOGE/USDT") == pytest.approx(0.03)
    assert step_size(f, p, "BTC/USDT") == pytest.approx(0.015)


def test_ladder_flat_strictly_below_bid_and_decreasing():
    lad = buy_ladder(100.0, None, 0.01, 5)
    assert len(lad) == 5
    assert all(p < 100.0 for p in lad)
    assert all(a > b for a, b in zip(lad, lad[1:]))
    assert lad[0] == pytest.approx(99.0) and lad[1] == pytest.approx(100 * 0.99**2)


def test_ladder_with_position_references_avg_entry_when_lower():
    lad = buy_ladder(100.0, 90.0, 0.01, 3)
    assert lad[0] == pytest.approx(90 * 0.99)
    assert all(p < 100.0 for p in lad)


def test_ladder_position_above_bid_uses_bid():
    lad = buy_ladder(100.0, 120.0, 0.01, 3)
    assert lad[0] == pytest.approx(99.0)


def test_take_profit_never_below_breakeven_plus_fees_falling_market():
    # Old C4 bug: price falls far below entry; TP must not chase the market down.
    p = _params()
    entry, ask = 100.0, 60.0
    step = 0.001  # tiny step so only the fee floor protects us
    tp = take_profit_price(entry, step, p, ask)
    assert tp >= entry * (1 + p.tp_fee_multiple * p.round_trip_fee)
    assert tp >= entry * (1 + step)


def test_take_profit_at_least_ask():
    p = _params()
    assert take_profit_price(100.0, 0.01, p, ask=130.0) == 130.0


def test_take_profit_uses_step_when_largest():
    p = _params()
    assert take_profit_price(100.0, 0.05, p, ask=101.0) == pytest.approx(105.0)
