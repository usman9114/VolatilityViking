from vv.core.regime import buys_allowed
from vv.core.types import SymbolFeatures


def _f(regime_ok: bool, warm: bool) -> SymbolFeatures:
    return SymbolFeatures(close=100.0, sigma_h=0.01, ema_4h=99.0, regime_ok=regime_ok, warm=warm)


def test_allows_when_warm_and_above_ema():
    assert buys_allowed(_f(True, True)) is True


def test_blocks_below_ema():
    assert buys_allowed(_f(False, True)) is False


def test_blocks_when_not_warm():
    assert buys_allowed(_f(True, False)) is False
    assert buys_allowed(_f(False, False)) is False
