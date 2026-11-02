import re

import pytest

from vv.core.ids import client_id


def test_deterministic():
    assert client_id("BTC/USDT", "buy", 3, 12345) == client_id("BTC/USDT", "buy", 3, 12345)


def test_stable_known_value():
    # Pinned: ids must not change across restarts, Python versions or
    # PYTHONHASHSEED. If this fails you changed the id scheme.
    assert client_id("BTC/USDT", "buy", 0, 0) == "vv-dab687a2b52c33f774172ba6c0f9f3e3"


def test_prefix_length_and_charset():
    cid = client_id("1000SHIB/USDT", "sell", 10, 2_000_000_000)
    assert cid.startswith("vv-")
    assert len(cid) <= 36
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,36}", cid)  # Binance newClientOrderId


def test_unique_per_level_side_symbol_bucket():
    ids = {
        client_id(sym, side, level, bucket)
        for sym in ("BTC/USDT", "SOL/USDT", "DOGE/USDT")
        for side in ("buy", "sell")
        for level in range(20)
        for bucket in (0, 1, 2)
    }
    assert len(ids) == 3 * 2 * 20 * 3


def test_symbol_separator_variants_do_not_collide_with_other_fields():
    assert client_id("BTC/USDT", "buy", 1, 23) != client_id("BTC/USDT", "buy", 12, 3)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        client_id("BTC/USDT", "hold", 0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        client_id("BTC/USDT", "buy", -1, 0)
    with pytest.raises(ValueError):
        client_id("BTC/USDT", "buy", 0, -1)
