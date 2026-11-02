"""Tests for research/carry_analysis.py (WP-S0E funding carry). No network, no data/."""
import numpy as np
import pandas as pd
import pytest

from research import carry_analysis as ca


def _grid(days: int, start="2024-01-01"):
    return pd.date_range(start, periods=days * 24, freq="1h") + pd.Timedelta(hours=1)


def _funding_grid(grid, daily_rates):
    """Hourly funding frame with 3 settlements/day (00:00, 08:00, 16:00)."""
    f = pd.Series(0.0, index=grid)
    for d, r in enumerate(daily_rates):
        base = grid[0].normalize() + pd.Timedelta(days=d)
        for h in (8, 16, 24):
            t = base + pd.Timedelta(hours=h)
            if t in f.index:
                f[t] = r
    return f


def test_annualisation_three_periods_per_day():
    idx = pd.date_range("2024-01-01", periods=3 * 30, freq="8h")
    r = pd.Series(0.0001, index=idx)
    assert ca.annualised_funding(r, days=30) == pytest.approx(0.0001 * 3 * 365)
    assert ca.PERIODS_PER_DAY == 3


def test_longest_negative_streak():
    r = pd.Series([0.1, -1, -1, 0.1, -1, -1, -1, 0.1])
    assert ca.longest_negative_streak(r) == 3


def test_exit_rule_fires_on_negative_run_and_reenters():
    grid = _grid(60)
    rates = [0.0003] * 20 + [-0.0003] * 15 + [0.0003] * 25
    on = ca.exit_state(_funding_grid(grid, rates))
    day = lambda d: int(d * 24)  # noqa: E731
    assert on[day(19)]                       # positive run: in
    assert not on[day(20 + 7)]               # 7 days into the negative run: out
    assert not on[day(34)]                   # still negative
    assert on[day(35 + 10)]                  # back in after funding recovers
    first_off = int(np.argmax(~on[day(10):])) + day(10)
    assert first_off < day(20 + 7)           # fired before the window fully turned


def test_exit_rule_does_not_use_future_funding():
    grid = _grid(40)
    base = [0.0003] * 40
    alt = [0.0003] * 20 + [-0.01] * 20
    a = ca.exit_state(_funding_grid(grid, base))
    b = ca.exit_state(_funding_grid(grid, alt))
    cut = 20 * 24
    assert np.array_equal(a[:cut], b[:cut])


def test_fees_charged_on_each_reentry():
    grid = _grid(70)
    rates = [0.0003] * 20 + [-0.0003] * 15 + [0.0003] * 35
    f = pd.DataFrame({"X": _funding_grid(grid, rates)})
    price = pd.DataFrame({"X": 100.0}, index=grid)
    res = ca.simulate(price, price, f)
    assert res.entries == 2 and res.exits == 1
    one_leg_fee = ca.SPOT_FRACTION * (ca.FEE_SPOT + ca.FEE_PERP)   # per entry or exit
    n_trades = res.entries + res.exits
    assert res.fees.sum() == pytest.approx(n_trades * one_leg_fee, rel=0.05)
    # each re-entry bar carries its own fee
    assert (res.fees > 0).sum() >= n_trades
    # equity = 1 + funding + mtm - fees (prices flat so mtm is 0)
    assert res.mtm.abs().sum() == 0
    assert res.equity.iloc[-1] == pytest.approx(
        1 + res.funding.sum() - res.fees.sum(), rel=1e-9)


def test_hold_always_pays_fee_once():
    grid = _grid(30)
    f = pd.DataFrame({"X": _funding_grid(grid, [0.0003] * 30)})
    price = pd.DataFrame({"X": 100.0}, index=grid)
    res = ca.simulate(price, price, f, use_exit=False)
    assert res.entries == 1 and res.exits == 0


def test_basis_mark_to_market_is_charged():
    grid = _grid(30)
    f = pd.DataFrame({"X": _funding_grid(grid, [0.0] * 30)})
    spot = pd.DataFrame({"X": 100.0}, index=grid)
    perp = spot.copy()
    perp.iloc[-1] = 101.0      # perp rallies 1% vs spot on the last bar: short loses
    res = ca.simulate(spot, perp, f, use_exit=False)
    assert res.mtm.iloc[-1] < 0
    assert res.mtm.iloc[-1] == pytest.approx(-0.5 * 1.0 / 100.0)


def _sel_inputs(n_days=120, seed=0):
    grid = _grid(n_days)
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        s: _funding_grid(grid, list(rng.normal(mu, 0.0001, n_days)))
        for s, mu in zip("ABCDE", [0.0004, 0.0003, 0.0002, 0.0001, 0.0])
    })
    return grid, f


def test_selection_uses_only_past_data():
    grid, f = _sel_inputs()
    avail = pd.DataFrame(True, index=grid, columns=f.columns)
    base = ca.monthly_selection(f, avail, k=3, window=7 * 24)

    cut = pd.Timestamp("2024-03-01 00:00")
    g = f.copy()
    after = g.index > cut
    # after the cut, reverse the symbol ranking (A <-> E, B <-> D)
    g.loc[after, ["A", "B", "D", "E"]] = g.loc[after, ["E", "D", "B", "A"]].to_numpy()
    shifted = ca.monthly_selection(g, avail, k=3, window=7 * 24)

    early = [t for t in base if t <= cut]
    assert early, "need month starts at or before the cut"
    for t in early:
        assert base[t] == shifted[t], t
    later = [t for t in base if t > cut + pd.Timedelta(days=8)]
    assert any(base[t] != shifted[t] for t in later)


def test_selection_ignores_data_just_after_month_start():
    grid, f = _sel_inputs()
    avail = pd.DataFrame(True, index=grid, columns=f.columns)
    base = ca.monthly_selection(f, avail, k=3, window=7 * 24)
    t0 = pd.Timestamp("2024-03-01 00:00")
    g = f.copy()
    g.loc[g.index > t0, :] = 0.0
    g.loc[g.index > t0, "E"] = 1.0
    alt = ca.monthly_selection(g, avail, k=3, window=7 * 24)
    assert base[t0] == alt[t0]


def test_selection_skips_unavailable_symbols():
    grid, f = _sel_inputs()
    avail = pd.DataFrame(True, index=grid, columns=f.columns)
    avail["A"] = False
    sel = ca.monthly_selection(f, avail, k=3, window=7 * 24)
    assert all("A" not in v for v in sel.values())


def test_max_drawdown_and_gate():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2])
    assert ca.max_drawdown(eq) == pytest.approx(0.1)
    yearly = pd.DataFrame({"days": [365, 365, 365], "raw": [0.05, 0.04, 0.03],
                           "annualised": [0.05, 0.04, 0.03]}, index=[2021, 2022, 2023])
    assert ca.gate_carry0(yearly, 0.05)["pass"]
    assert not ca.gate_carry0(yearly, 0.09)["pass"]
    yearly.loc[2022, "annualised"] = -0.03
    assert not ca.gate_carry0(yearly, 0.05)["pass"]


def test_margin_event_when_perp_doubles_between_rebalances():
    grid = _grid(20)
    f = pd.DataFrame({"X": _funding_grid(grid, [0.0003] * 20)})
    spot = pd.DataFrame({"X": 100.0}, index=grid)
    ramp = np.r_[np.full(100, 100.0), np.linspace(100, 210, 100), np.full(280, 210.0)]
    spot = pd.DataFrame({"X": ramp}, index=grid)
    perp = spot.copy()               # both double: delta-neutral but short perp margin is gone
    res = ca.simulate(spot, perp, f, use_exit=False)
    assert res.margin_events == 1
    # a drift-triggered resize tops the sleeve back up and avoids the event
    res2 = ca.simulate(spot, perp, f, use_exit=False, drift_tol=0.25)
    assert res2.margin_events == 0
    assert res2.fees.sum() > res.fees.sum()
