from decimal import Decimal

import pytest

from vv.core.filters import round_price, round_qty, validate_notional
from vv.core.types import SymbolFilters

F = SymbolFilters(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=10.0)


def test_buy_price_rounds_down():
    assert round_price("buy", 100.019, 0.01) == 100.01


def test_sell_price_rounds_up():
    assert round_price("sell", 100.011, 0.01) == 100.02


def test_price_on_tick_unchanged_both_sides():
    assert round_price("buy", 100.01, 0.01) == 100.01
    assert round_price("sell", 100.01, 0.01) == 100.01


def test_price_decimal_edge_0_1_plus_0_2():
    # 0.1 + 0.2 == 0.30000000000000004 in binary floats; a float floor/ceil
    # implementation would push the sell up a full tick.
    p = 0.1 + 0.2
    assert round_price("sell", p, 0.1) == 0.3
    assert round_price("buy", p, 0.1) == 0.3


def test_price_float_tick_drift_does_not_move_exact_multiple():
    # 0.7 / 0.1 == 6.999999999999999 as floats; floor would give 0.6.
    assert round_price("buy", 0.7, 0.1) == 0.7
    assert round_price("sell", 0.7, 0.1) == 0.7


def test_qty_always_rounds_down():
    assert round_qty(0.0019, 0.001) == 0.001
    assert round_qty(0.00099, 0.001) == 0.0
    assert round_qty(1.0, 0.001) == 1.0


def test_qty_decimal_edge_0_1_plus_0_2():
    assert round_qty(0.1 + 0.2, 0.1) == 0.3


def test_round_returns_float():
    assert isinstance(round_price("buy", 1.234, 0.01), float)
    assert isinstance(round_qty(1.234, 0.01), float)


def test_invalid_side_and_nonpositive_increment_rejected():
    with pytest.raises(ValueError):
        round_price("hold", 1.0, 0.01)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        round_price("buy", 1.0, 0.0)
    with pytest.raises(ValueError):
        round_qty(1.0, -0.1)


def test_notional_ok_after_rounding():
    price = round_price("buy", 100.019, F.tick_size)
    qty = round_qty(0.1019, F.step_size)
    assert validate_notional(price, qty, F)


def test_min_notional_rejected_after_rounding():
    # Raw notional is 10.0009 (passes); rounding price and qty down leaves
    # 100.00 * 0.099 = 9.9 (fails).
    raw_price, raw_qty = 100.009, 0.1000
    assert raw_price * raw_qty >= F.min_notional
    price = round_price("buy", raw_price, F.tick_size)  # 100.00
    qty = round_qty(0.09999, F.step_size)  # 0.099
    assert not validate_notional(price, qty, F)


def test_notional_exact_boundary_passes_using_decimal():
    # 0.3 * 3.0 == 0.8999999999999999 in floats, 0.9 exactly in Decimal:
    # a float compare against min_notional=0.9 would wrongly reject.
    assert 0.3 * 3.0 < 0.9
    f = SymbolFilters(tick_size=0.1, step_size=0.1, min_qty=0.1, min_notional=0.9)
    assert validate_notional(0.3, 3.0, f)
    assert Decimal("0.3") * Decimal("3.0") == Decimal("0.9")


def test_min_qty_rejected():
    assert not validate_notional(1000.0, 0.0005, F)


def test_zero_qty_rejected():
    assert not validate_notional(100.0, 0.0, F)
