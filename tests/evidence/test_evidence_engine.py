import hashlib
import math

import numpy as np
import pytest

from vv.evidence.decay import CusumMonitor
from vv.evidence.posterior import hierarchical_prior, p_positive, posterior
from vv.evidence.registry import (
    PreregMismatch,
    Registry,
    Signal,
    append_ledger,
    load_ledger,
    verify_prereg,
)
from vv.evidence.weights import factor_budget, incremental_alpha, kelly_weights, ramp_cap


def test_posterior_matches_hand_computation():
    prior = (0.0, 0.5)
    sr, se = 0.77, 0.7
    mean, sd = posterior(prior, (sr, se), None)
    p0, p1 = 1 / 0.25, 1 / se**2
    assert mean == pytest.approx((sr * p1) / (p0 + p1))
    assert sd == pytest.approx(math.sqrt(1 / (p0 + p1)))


def test_posterior_no_evidence_returns_prior():
    assert posterior((0.1, 0.5), None, None) == pytest.approx((0.1, 0.5))


def test_posterior_forward_update_and_eff_n():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 730)
    m_full, s_full = posterior((0.0, 0.5), None, r)
    m_eff, s_eff = posterior((0.0, 0.5), None, r, eff_n_factor=0.25)
    assert s_eff > s_full  # less independent info -> wider
    sr = r.mean() / r.std(ddof=1) * math.sqrt(365)
    se = math.sqrt((1 + 0.5 * sr**2) / (730 / 365))
    p0, p1 = 4.0, 1 / se**2
    assert m_full == pytest.approx(sr * p1 / (p0 + p1))


def test_p_positive():
    assert p_positive(0.0, 1.0) == pytest.approx(0.5)
    assert p_positive(1.0, 1.0) == pytest.approx(0.8413, abs=1e-3)


def test_s1_worked_example():
    mean, sd = posterior((0.0, 0.5), (0.77, math.sqrt((1 + 0.5 * 0.77**2) / 2.83)), None)
    assert 0.2 < mean < 0.3
    p = p_positive(mean, sd)
    assert 0.70 < p < 0.78
    assert kelly_weights({"S1": (mean, sd)}, {"S1": 0.1})["S1"] == 0.0


def test_hierarchical_prior_fallback_and_shrinkage():
    assert hierarchical_prior([1, 2], [0.1, 0.1]) == (0.0, 0.5)
    srs = [0.2, 0.4, 0.3, 0.5, 0.1, 0.35]
    ses = [0.3] * 6
    mu, tau = hierarchical_prior(srs, ses)
    assert mu == pytest.approx(np.mean(srs))
    assert tau < np.std(srs, ddof=1)  # noise variance removed
    # a high T1 estimate is shrunk toward the ledger mean
    m, _ = posterior((mu, tau), (1.5, 0.3), None)
    assert mu < m < 1.5


def test_kelly_zero_below_min_p():
    post = {"a": (0.2, 0.5), "b": (0.6, 0.2)}  # a: P~0.65 ; b: P~0.9987
    w = kelly_weights(post, {"a": 0.1, "b": 0.1})
    assert w["a"] == 0.0
    assert w["b"] == pytest.approx(0.25 * 0.6 / 0.1)


def test_factor_budget_caps_sum():
    w = {"a": 0.3, "b": 0.3, "c": 0.1, "d": 0.05}
    out = factor_budget(w, {"a": "risk_on", "b": "risk_on", "c": "carry", "d": "carry"},
                        {"risk_on": 0.4, "carry": 0.5})
    assert out["a"] + out["b"] == pytest.approx(0.4)
    assert out["a"] == pytest.approx(out["b"])
    assert out["c"] == 0.1 and out["d"] == 0.05  # under budget: untouched


def test_ramp_cap_monotone_and_bounded():
    vals = [ramp_cap(0.05, p, 1.0) for p in (0.5, 1, 2, 4, 100, 1e6)]
    assert vals == sorted(vals)
    assert max(vals) == 1.0
    assert ramp_cap(0.05, 1.0, 1.0) == pytest.approx(0.05)
    assert ramp_cap(0.05, 4.0, 1.0) == pytest.approx(0.10)


def test_incremental_alpha_recovers_residual():
    rng = np.random.default_rng(3)
    book = rng.normal(0, 0.01, 2000)
    sig = 0.8 * book + rng.normal(0.001, 0.005, 2000)
    a, se = incremental_alpha(sig, book)
    assert a > 0 and se > 0


def _run(mon, xs):
    states = [mon.update(x) for x in xs]
    return states


def test_cusum_regime_break_ok_halve_zero():
    rng = np.random.default_rng(11)
    mon = CusumMonitor(expected_daily_mean=0.001, sd=0.01)
    good = rng.normal(0.001, 0.01, 60)
    assert set(_run(mon, good)) == {"ok"}
    bad = rng.normal(-0.01, 0.01, 200)
    states = _run(mon, bad)
    assert "halve" in states and states[-1] == "zero"
    assert states.index("halve") < states.index("zero")


def test_cusum_stationary_false_alarm_rate():
    # 2 years of daily data; alarm = any breach. Use h=8 (ARL0 >> 730).
    rng = np.random.default_rng(42)
    n_paths, alarms = 400, 0
    for _ in range(n_paths):
        mon = CusumMonitor(0.0005, 0.01, k=0.5, h=8)
        if any(s != "ok" for s in _run(mon, rng.normal(0.0005, 0.01, 730))):
            alarms += 1
    assert alarms / n_paths < 0.05


def test_prereg_hash_mismatch_raises(tmp_path):
    f = tmp_path / "pre.json"
    f.write_text('{"a": 1}')
    good = hashlib.sha256(f.read_bytes()).hexdigest()
    kw = dict(id="X", family="f", role="directional", hypothesis="h", sign=1, horizon="1d",
              data="d", prereg_path="pre.json", status="candidate", created="2026-11-17",
              t1={"sr_oos_stress": 0.5, "se": 0.5, "years": 2.0})
    reg = Registry(signals=[Signal(prereg_sha256=good, **kw)])
    verify_prereg(reg, tmp_path)
    f.write_text('{"a": 2}')
    with pytest.raises(PreregMismatch):
        verify_prereg(reg, tmp_path)


def test_ledger_append_only(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("id,date,family,outcome,sr,se\n")
    append_ledger(p, id="a", date="2026-11-17", family="f", outcome="fail")
    with pytest.raises(ValueError):
        append_ledger(p, id="a", date="2026-11-17", family="f", outcome="fail")
    assert [r["id"] for r in load_ledger(p)] == ["a"]
