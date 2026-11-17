import logging
import random

import pandas as pd
import pytest

from tests.runner.conftest import NOW, SYMS, factory_for, fake_venue, make_cfg, make_fetcher, write_repo
from vv.exec.common import OrderResult
from vv.runner import core
from vv.runner.alerts import Alerter, install_log_redaction, redact
from vv.runner.config import Caps, venue_symbol
from vv.runner.engine import apply_caps, combine_book
from vv.runner.state import State

LAST = pd.Timestamp("2026-06-09", tz="UTC")


def run(cfg, frame, mode="dry", **kw):
    kw.setdefault("fetch", make_fetcher(frame))
    kw.setdefault("env", {})
    kw.setdefault("alerter", Alerter(kw["env"], http_get=lambda u: None))
    return core.run(cfg, mode, now=NOW, **kw)


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path, write_repo(tmp_path))


# ---- gaps -------------------------------------------------------------------
def test_missing_bar_skips_trading_alerts_and_exits_nonzero(tmp_path, frame):
    cfg = make_cfg(tmp_path, write_repo(tmp_path))
    v = fake_venue(frame)
    fac = factory_for(v)
    r = run(cfg, frame, "testnet", fetch=make_fetcher(frame, drop=("BBBUSDT", LAST)), adapter_factory=fac)
    assert r.code == core.GAP != 0
    assert v.placed == [] and fac.made == []
    assert any("DATA GAP" in a and "BBBUSDT" in a for a in r.alerts)
    assert (tmp_path / "reports/2026-06-10.md").exists()


def test_hole_in_history_is_a_gap(tmp_path, frame):
    cfg = make_cfg(tmp_path, write_repo(tmp_path))
    hole = LAST - pd.Timedelta(days=40)
    r = run(cfg, frame, fetch=make_fetcher(frame, drop=("AAAUSDT", hole)))
    assert r.code == core.GAP and any("missing bar" in a for a in r.alerts)


def test_stale_data_is_a_gap(tmp_path, frame):
    cfg = make_cfg(tmp_path, write_repo(tmp_path))
    late = pd.Timestamp("2026-06-11 06:00", tz="UTC")  # bars end at 06-10 00:00: 30h old, fetch returns nothing new
    short = frame.loc[:LAST]
    r = core.run(cfg, "dry", now=late, fetch=make_fetcher(short), env={},
                 alerter=Alerter({}, http_get=lambda u: None))
    assert r.code == core.GAP


# ---- halt -------------------------------------------------------------------
def test_halt_blocks_next_run_and_resume_clears(tmp_path, frame, cfg):
    st = State(cfg.state_db)
    st.set_halt("test halt")
    st.close()
    r = run(cfg, frame)
    assert r.code == core.HALTED and any("HALTED" in a for a in r.alerts)
    st = State(cfg.state_db)
    assert st.halt_active()
    assert st.resume("operator checked") == 1
    assert not st.halt_active()
    row = st.db.execute("SELECT active, clear_reason FROM halt").fetchone()
    assert (row["active"], row["clear_reason"]) == (0, "operator checked")
    st.close()
    assert run(cfg, frame).code == core.OK


def test_resume_cli(tmp_path, cfg):
    import yaml

    from vv.runner.__main__ import main
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump()))
    st = State(cfg.state_db)
    st.set_halt("x")
    st.close()
    assert main(["resume", "--reason", "ok now", "--config", str(p)]) == 0
    st = State(cfg.state_db)
    assert not st.halt_active()


# ---- live guard -------------------------------------------------------------
def test_live_without_confirm_refuses(cfg, frame):
    v = fake_venue(frame)
    fac = factory_for(v)
    r = run(cfg, frame, "live", adapter_factory=fac, env={})
    assert r.code == core.LIVE_REFUSED
    assert fac.made == [] and v.placed == []
    r = run(cfg, frame, "live", adapter_factory=fac, env={"VV_LIVE_CONFIRM": "no"})
    assert r.code == core.LIVE_REFUSED and fac.made == []


def test_live_with_confirm_proceeds_on_fake_venue(cfg, frame):
    fac = factory_for(fake_venue(frame))
    r = run(cfg, frame, "live", adapter_factory=fac, env={"VV_LIVE_CONFIRM": "yes"})
    assert r.code == core.OK and fac.made == ["live"]


# ---- caps -------------------------------------------------------------------
def test_apply_caps_random():
    rng = random.Random(7)
    caps = Caps(gross=1.0, short=0.5, per_coin=0.25)
    for _ in range(500):
        w = pd.Series({s: rng.uniform(-2, 2) for s in SYMS})
        out = apply_caps(w, caps, allow_short=True)
        assert out.abs().max() <= 0.25 + 1e-9
        assert out.abs().sum() <= 1.0 + 1e-9
        assert -out[out < 0].sum() <= 0.5 + 1e-9
        assert (apply_caps(w, caps, allow_short=False) >= 0).all()


def test_combined_book_respects_caps_end_to_end(tmp_path, frame):
    root = write_repo(tmp_path, ids=("P1", "P2"))
    cfg = make_cfg(tmp_path, root, caps=Caps(gross=1.0, short=0.2, per_coin=0.25), statuses=["live"])
    r = run(cfg, frame)
    assert r.code == core.OK
    book = pd.Series(r.book)
    assert any(e.weight > 0 for e in r.evidence)
    assert book.abs().max() <= 0.25 + 1e-9 and book.abs().sum() <= 1.0 + 1e-9
    assert -book[book < 0].sum() <= 0.2 + 1e-9
    assert (book < 0).any() and (book > 0).any()  # synthetic data trends both ways


def test_combine_book_scales_big_weights_down():
    t = {"A": pd.Series({s: 0.25 for s in SYMS}), "B": pd.Series({s: -0.25 for s in SYMS})}
    out = combine_book(t, {"A": 5.0, "B": 2.0}, SYMS, Caps(), allow_short=True)
    assert out.abs().sum() <= 1.0 + 1e-9 and out.abs().max() <= 0.25 + 1e-9


# ---- zero weight ------------------------------------------------------------
def test_zero_evidence_weight_means_no_orders(tmp_path, frame):
    cfg = make_cfg(tmp_path, write_repo(tmp_path, t1=(0.3, 0.7)))
    v = fake_venue(frame)
    r = run(cfg, frame, "testnet", adapter_factory=factory_for(v))
    assert r.code == core.OK
    assert all(e.weight == 0.0 for e in r.evidence)
    assert all(w == 0.0 for w in r.book.values())
    assert r.orders == [] and v.placed == []


def test_candidate_only_registry_is_a_noop(tmp_path, frame):
    cfg = make_cfg(tmp_path, write_repo(tmp_path, status="candidate"))
    r = run(cfg, frame)
    assert r.code == core.OK and r.evidence == [] and r.orders == []


# ---- reconcile --------------------------------------------------------------
def test_reconcile_diff_halts(tmp_path, frame, cfg):
    v = fake_venue(frame)
    v.qty[venue_symbol(SYMS[0], "perp")] = 1.5  # position the state store knows nothing about
    v.entry[venue_symbol(SYMS[0], "perp")] = 100.0
    for sy in SYMS:
        v.margin_mode[venue_symbol(sy, "perp")] = "isolated"
        v.leverage[venue_symbol(sy, "perp")] = 1
    r = run(cfg, frame, "testnet", adapter_factory=factory_for(v))
    assert r.code == core.RECONCILE_HALT
    assert v.placed == [] and any("reconcile" in a for a in r.alerts)
    st = State(cfg.state_db)
    assert st.halt_active()
    st.close()
    assert run(cfg, frame, "testnet", adapter_factory=factory_for(v)).code == core.HALTED


# ---- determinism / idempotency ---------------------------------------------
def test_dry_run_is_deterministic(tmp_path, frame, cfg):
    r1 = run(cfg, frame)
    r2 = run(cfg, frame)
    assert r1.orders, "synthetic trends should yield orders"
    assert [(o.client_id, o.symbol, o.side, o.qty) for o in r1.orders] == \
           [(o.client_id, o.symbol, o.side, o.qty) for o in r2.orders]
    assert all(isinstance(o, OrderResult) and o.status == "dry_run" for o in r1.orders)


def test_testnet_rerun_places_nothing_new(tmp_path, frame, cfg):
    v = fake_venue(frame)
    r1 = run(cfg, frame, "testnet", adapter_factory=factory_for(v))
    assert r1.code == core.OK and v.placed
    n = len(v.placed)
    r2 = run(cfg, frame, "testnet", adapter_factory=factory_for(v))
    assert r2.code == core.OK and len(v.placed) == n
    st = State(cfg.state_db)
    assert st.db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == n
    assert st.positions()


# ---- evidence plumbing ------------------------------------------------------
def test_shadow_pnl_accumulates_over_consecutive_days(tmp_path, frame):
    cfg = make_cfg(tmp_path, write_repo(tmp_path))
    for d in range(3):
        now = NOW - pd.Timedelta(days=2 - d)
        core.run(cfg, "dry", now=now, fetch=make_fetcher(frame.loc[:now.normalize() - pd.Timedelta(days=1)]),
                 env={}, alerter=Alerter({}, http_get=lambda u: None))
    st = State(cfg.state_db)
    assert len(st.pnl_series("P1")) == 2
    r = run(cfg, frame)
    assert r.evidence[0].fwd_n == 2
    # perp signal with no funding archive: estimated 0 and flagged
    assert st.db.execute("SELECT MIN(funding_missing) FROM shadow_pnl").fetchone()[0] == 1


def test_dial_without_external_data_alerts_and_applies_no_penalty(tmp_path, frame):
    cfg = make_cfg(tmp_path, write_repo(tmp_path, ids=("P2",)))
    r = run(cfg, frame)
    assert r.code == core.OK and any("dial inputs stale/missing" in a for a in r.alerts)


# ---- secrets ----------------------------------------------------------------
def test_redaction_everywhere(tmp_path, frame, caplog):
    secret = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ_0123"
    env = {"VV_TELEGRAM_TOKEN": secret, "VV_TELEGRAM_CHAT": "99887766", "VV_BINANCE_SECRET": "s3cr3t-value-xyz",
           "VV_HEARTBEAT_URL": "https://hc-ping.com/abcd-1234-uuid"}
    assert secret not in redact(f"url bot{secret}/send {secret}", env)
    assert "s3cr3t-value-xyz" not in redact("x s3cr3t-value-xyz y", env)
    assert "bot123456789:ABC" not in redact("https://api.telegram.org/bot" + secret + "/x", {})

    sent = []

    def boom(url):
        sent.append(url)
        raise RuntimeError("failed calling " + url)  # exception text embeds the token

    cfg = make_cfg(tmp_path, write_repo(tmp_path))
    install_log_redaction(env)
    with caplog.at_level(logging.DEBUG):
        a = Alerter(env, http_get=boom)
        a.alert(f"something with {secret} and s3cr3t-value-xyz")
        a.heartbeat()
        logging.getLogger("vv.runner").warning("leak? %s", secret)
        r = core.run(cfg, "dry", now=NOW, fetch=make_fetcher(frame), env=env, alerter=a)
    assert sent, "telegram was attempted"
    blob = "\n".join(rec.getMessage() for rec in caplog.records) + "".join(a.messages) + \
        (tmp_path / "reports/2026-06-10.md").read_text()
    assert secret not in blob and "s3cr3t-value-xyz" not in blob and "hc-ping.com/abcd" not in blob
    assert r.code == core.OK


def test_no_telegram_env_means_log_only(frame):
    seen = []
    a = Alerter({}, http_get=seen.append)
    a.alert("hello")
    a.heartbeat()
    assert seen == [] and a.messages == ["hello"]


def test_heartbeat_posted_when_url_set():
    seen = []
    a = Alerter({"VV_HEARTBEAT_URL": "https://hc.example/ping"}, http_get=seen.append)
    a.heartbeat(True)
    a.heartbeat(False)
    assert seen == ["https://hc.example/ping", "https://hc.example/ping/fail"]
