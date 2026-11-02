import pytest

from vv.core.params import RiskParams
from vv.core.sizing import budget_remaining, ladder_quantities, symbol_budgets

RISK = RiskParams(
    total_exposure_cap=0.5, per_symbol_cap=0.3, portfolio_halt_dd=0.2, symbol_halt_dd=0.15
)


def test_budgets_inverse_vol_split_sums_to_pool():
    b = symbol_budgets(1000.0, {"A": 0.01, "B": 0.02}, RISK)
    # 500 pool, shares 2/3 and 1/3 -> 333 capped at 300, 167
    assert b["A"] == pytest.approx(300.0)
    assert b["B"] == pytest.approx(500 / 3)
    assert sum(b.values()) <= 0.5 * 1000 + 1e-9


def test_budgets_equal_sigma_equal_split_under_cap():
    b = symbol_budgets(1000.0, {"A": 0.01, "B": 0.01, "C": 0.01}, RISK)
    assert all(v == pytest.approx(500 / 3) for v in b.values())


def test_budgets_per_symbol_cap_applies_to_single_symbol():
    assert symbol_budgets(1000.0, {"A": 0.01}, RISK) == {"A": pytest.approx(300.0)}


def test_budgets_nonpositive_equity_or_empty():
    assert symbol_budgets(0.0, {"A": 0.01}, RISK) == {"A": 0.0}
    assert symbol_budgets(-5.0, {"A": 0.01}, RISK) == {"A": 0.0}
    assert symbol_budgets(1000.0, {}, RISK) == {}


def test_budgets_zero_sigma_is_finite():
    b = symbol_budgets(1000.0, {"A": 0.0, "B": 0.01}, RISK)
    assert all(v == v and v >= 0 for v in b.values())


def test_ladder_notional_sums_to_budget_and_weights_geometric():
    prices = [100.0, 98.0, 96.0]
    q = ladder_quantities(prices, 300.0, 2.0)
    notionals = [p * x for p, x in zip(prices, q)]
    assert sum(notionals) == pytest.approx(300.0)
    assert notionals[1] / notionals[0] == pytest.approx(2.0)
    assert notionals[2] / notionals[1] == pytest.approx(2.0)


def test_ladder_zero_or_negative_budget_gives_zero_quantities():
    assert ladder_quantities([1.0, 2.0], 0.0, 1.5) == [0.0, 0.0]
    assert ladder_quantities([1.0, 2.0], -10.0, 1.5) == [0.0, 0.0]
    assert ladder_quantities([], 100.0, 1.5) == []


def test_budget_remaining_never_negative():
    assert budget_remaining(100.0, 30.0) == 70.0
    assert budget_remaining(100.0, 130.0) == 0.0
