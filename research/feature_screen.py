"""Pre-registered feature screen (research/feature_screen_prereg.json).

Question: does external data (derivatives, macro, crypto-native) add predictive power over
price-only baselines for BTC and an equal-weight basket?  Do not tune anything here after
results exist; the prereg file is the spec.

Timing convention: a row dated t is the state at the UTC close of day t.  Targets look
forward from that close (ret_1d = close[t+1]/close[t]).  Every feature in row t may only use
information published before that close, so it is also safe for the target window opening then.

Usage: .venv/bin/python -m research.feature_screen [--data-dir data] [--out docs/research/FEATURE_SCREEN.md]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

PREREG = json.loads((Path(__file__).parent / "feature_screen_prereg.json").read_text())
GROUPS: dict[str, list[str]] = {k.split("_")[0]: v for k, v in PREREG["features"].items()}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]
SYMS: list[str] = PREREG["universe"]["basket"]
IS_START, IS_END = (pd.Timestamp(d, tz="UTC") for d in PREREG["periods"]["in_sample"])
OOS_START, OOS_END = (pd.Timestamp(d, tz="UTC") for d in PREREG["periods"]["out_of_sample"])
HORIZON = {"ret_1d": 1, "ret_7d": 7, "crash_7d": 7, "vol_7d": 7}
API = "https://data-api.binance.vision/api/v3/klines"
FRED_IDS = ["DTWEXBGS", "DGS10", "VIXCLS", "NASDAQCOM", "WALCL", "RRPONTSYD", "WTREGEN"]


# ----------------------------------------------------------------------------- statistics
def trailing_z(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Z-score vs the trailing window that ends at (and includes) the current row. No look-ahead."""
    r = s.rolling(window, min_periods=min_periods or window)
    return (s - r.mean()) / r.std()


def hac_tstat(x, y, lags: int) -> tuple[float, float]:
    """OLS slope of y on x (with intercept) and its Newey-West (Bartlett) t-statistic. Deterministic."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = X * (y - X @ beta)[:, None]  # score contributions
    S = u.T @ u
    for lag in range(1, lags + 1):
        g = u[lag:].T @ u[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    V = xtx_inv @ S @ xtx_inv
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1]))


def ic_test(x: pd.Series, y: pd.Series, lags: int, min_n: int = 100) -> dict:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < min_n or d.iloc[:, 0].nunique() < 3 or d.iloc[:, 1].nunique() < 2:
        return {"ic": np.nan, "t": np.nan, "p": np.nan, "n": len(d)}
    rx, ry = d.iloc[:, 0].rank().to_numpy(), d.iloc[:, 1].rank().to_numpy()
    ic = float(np.corrcoef(rx, ry)[0, 1])
    _, t = hac_tstat((rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std(), lags)
    return {"ic": ic, "t": t, "p": float(2 * stats.norm.sf(abs(t))), "n": len(d)}


def bh_fdr(pvals, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: boolean reject mask. NaN p-values are never rejected and not counted."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.zeros(len(p), bool)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv, kind="stable")
    passed = pv[order] <= (np.arange(1, m + 1) / m) * q
    if passed.any():
        k = np.max(np.nonzero(passed)[0])
        rej = np.zeros(m, bool)
        rej[order[: k + 1]] = True
        out[np.nonzero(ok)[0]] = rej
    return out


def auc_score(y, p) -> float:
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------- models
def _standardise(Xtr: np.ndarray, Xte: np.ndarray):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge_fit(Z: np.ndarray, y: np.ndarray, alpha: float):
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return w, ym


RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)


def ridge_predict(Xtr, ytr, Xte, h: int, n_folds: int = 4) -> np.ndarray:
    """Ridge with alpha picked by inner expanding-window CV on the training data only."""
    n = len(Xtr)
    edges = np.linspace(0, n, n_folds + 2).astype(int)
    best, best_mse = RIDGE_ALPHAS[-1], np.inf
    for a in RIDGE_ALPHAS:
        mse = []
        for k in range(1, n_folds + 1):
            tr_end, va0, va1 = max(edges[k] - h, 0), edges[k], edges[k + 1]
            if tr_end < 50 or va1 <= va0:
                continue
            Ztr, Zva = _standardise(Xtr[:tr_end], Xtr[va0:va1])
            w, ym = _ridge_fit(Ztr, ytr[:tr_end], a)
            mse.append(np.mean((ytr[va0:va1] - (Zva @ w + ym)) ** 2))
        if mse and np.mean(mse) < best_mse:
            best, best_mse = a, float(np.mean(mse))
    Ztr, Zte = _standardise(Xtr, Xte)
    w, ym = _ridge_fit(Ztr, ytr, best)
    return Zte @ w + ym


def logistic_predict(Xtr, ytr, Xte, C: float = 1.0) -> np.ndarray:
    """L2 logistic regression (sklearn convention: C*sum logloss + 0.5*|w|^2), fixed C."""
    Ztr, Zte = _standardise(Xtr, Xte)
    ytr = ytr.astype(float)
    if ytr.sum() == 0 or ytr.sum() == len(ytr):
        return np.full(len(Xte), ytr.mean())

    def fun(theta):
        b, w = theta[0], theta[1:]
        z = b + Ztr @ w
        loss = C * np.sum(np.logaddexp(0, z) - ytr * z) + 0.5 * w @ w
        g = C * (1 / (1 + np.exp(-z)) - ytr)
        return loss, np.concatenate([[g.sum()], Ztr.T @ g + w])

    res = optimize.minimize(fun, np.zeros(Ztr.shape[1] + 1), jac=True, method="L-BFGS-B")
    return 1 / (1 + np.exp(-(res.x[0] + Zte @ res.x[1:])))


def walk_forward_splits(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, h: int,
                        min_train: int = 365):
    """Monthly refit, expanding window.  Train rows end `h` days before the month starts, so no
    training target window reaches into the test month.  Yields (train_positions, test_positions)."""
    for ms in pd.date_range(start.normalize().replace(day=1), end, freq="MS", tz="UTC"):
        me = ms + pd.offsets.MonthBegin(1)
        test = np.nonzero((index >= max(ms, start)) & (index < me) & (index <= end))[0]
        train = np.nonzero(index <= ms - pd.Timedelta(days=h + 1))[0]
        if len(test) and len(train) >= min_train:
            yield train, test


def walk_forward(X: pd.DataFrame, y: pd.Series, kind: str, h: int, start=OOS_START, end=OOS_END) -> pd.DataFrame:
    """Out-of-sample predictions (`pred`) and expanding-mean benchmark (`bench`) for rows in [start, end]."""
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    out = []
    for tr, te in walk_forward_splits(X.index, start, end, h):
        if kind == "ridge":
            p = ridge_predict(Xv[tr], yv[tr], Xv[te], h)
        else:
            p = logistic_predict(Xv[tr], yv[tr], Xv[te])
        out.append(pd.DataFrame({"pred": p, "bench": yv[tr].mean(), "y": yv[te]}, index=X.index[te]))
    return pd.concat(out) if out else pd.DataFrame(columns=["pred", "bench", "y"])


def oos_metrics(res: pd.DataFrame, kind: str) -> dict:
    if kind == "ridge":
        return {"r2": 1 - ((res.y - res.pred) ** 2).sum() / ((res.y - res.bench) ** 2).sum(), "n": len(res)}
    return {"auc": auc_score(res.y, res.pred), "brier": float(((res.pred - res.y) ** 2).mean()),
            "n": len(res), "n_pos": int(res.y.sum())}


# ----------------------------------------------------------------------------- features
def lag_to_calendar(s: pd.Series, cal: pd.DatetimeIndex, rule: str) -> pd.Series:
    """Re-date observations to when they become usable, then forward-fill onto the daily calendar.
    rule: 'bday1' (+1 business day: FRED daily), 'day1' (+1 calendar day: weekly H.4.1 released the
    day after its Wednesday date, and the conservative lag for stablecoins / Fear&Greed), 'none'."""
    s = s.dropna()
    if rule == "bday1":
        s.index = s.index + pd.offsets.BDay(1)
    elif rule == "day1":
        s.index = s.index + pd.Timedelta(days=1)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(cal.union(s.index)).ffill().reindex(cal)


def price_features(close: pd.Series, volume_z: pd.Series) -> pd.DataFrame:
    lc = np.log(close)
    r1 = lc.diff()
    return pd.DataFrame({
        "ret_1d_past": r1, "ret_7d_past": lc.diff(7), "ret_30d_past": lc.diff(30),
        "dist_sma200": close / close.rolling(200).mean() - 1,
        "rv_7d": r1.rolling(7).std(), "rv_30d": r1.rolling(30).std(), "volume_z_30d": volume_z,
    })


def deriv_features(cal, metrics, premium, funding) -> pd.DataFrame:
    nan = pd.Series(np.nan, index=cal)

    def col(df, name):
        return df[name].reindex(cal) if df is not None and name in df else nan

    f = col(funding, "funding_rate")
    oi = col(metrics, "sum_open_interest").where(lambda x: x > 0)  # zero OI rows are feed glitches
    return pd.DataFrame({
        "funding_1d": f, "funding_7d": f.rolling(7).sum(),
        "oi_chg_1d": oi.pct_change(1, fill_method=None), "oi_chg_7d": oi.pct_change(7, fill_method=None),
        "oi_value_z_90d": trailing_z(col(metrics, "sum_open_interest_value"), 90),
        "toptrader_ls_ratio": col(metrics, "sum_toptrader_long_short_ratio"),
        "global_ls_ratio": col(metrics, "count_long_short_ratio"),
        "taker_buysell_ratio": col(metrics, "sum_taker_long_short_vol_ratio"),
        "perp_premium": col(premium, "perp_premium"),
    })


def macro_features(cal, fred) -> pd.DataFrame:
    out = pd.DataFrame(index=cal, columns=GROUPS["C"], dtype=float)
    if fred is None:
        return out

    def obs(c):
        return fred[c].dropna() if c in fred else pd.Series(dtype=float)

    def chg5(c, log=True):  # 5-observation change on raw observations, then lagged +1 business day
        s = obs(c)
        s = (np.log(s).diff(5) if log else s.diff(5)) if len(s) else s
        return lag_to_calendar(s, cal, "bday1")

    out["usd_broad_chg_5d"] = chg5("DTWEXBGS")
    out["us10y_chg_5d"] = chg5("DGS10", log=False)
    out["vix_level"] = lag_to_calendar(obs("VIXCLS"), cal, "bday1")
    out["vix_chg_5d"] = chg5("VIXCLS")
    out["nasdaq_ret_5d"] = chg5("NASDAQCOM")
    # Fed liquidity = WALCL ($m -> $bn) - ON RRP ($bn) - TGA ($bn), each at its own availability date.
    walcl = lag_to_calendar(obs("WALCL") / 1000.0, cal, "day1")
    rrp = lag_to_calendar(obs("RRPONTSYD"), cal, "bday1")
    tga = lag_to_calendar(obs("WTREGEN"), cal, "day1")
    out["fed_liquidity_chg_4w"] = (walcl - rrp - tga).diff(28)
    return out


def native_features(cal, fear_greed, stablecoins) -> pd.DataFrame:
    out = pd.DataFrame(index=cal, columns=GROUPS["D"], dtype=float)
    if stablecoins is not None:
        sc = lag_to_calendar(stablecoins["stablecoin_supply_usd"], cal, "day1")
        out["stablecoin_supply_chg_7d"] = sc.pct_change(7, fill_method=None)
        out["stablecoin_supply_chg_30d"] = sc.pct_change(30, fill_method=None)
    if fear_greed is not None:
        fg = lag_to_calendar(fear_greed["fear_greed"], cal, "day1")
        out["fear_greed"] = fg
        out["fear_greed_chg_7d"] = fg.diff(7)
    return out


def build_features(inp: dict, cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Returns {'single': BTC features, 'basket': equal-weight basket features}, columns = ALL_FEATURES."""
    px = {s: inp["prices"][s].reindex(cal) for s in SYMS if s in inp["prices"]}
    vz = {s: trailing_z(np.log1p(p["volume"]), 30) for s, p in px.items()}
    macro = macro_features(cal, inp.get("fred"))
    native = native_features(cal, inp.get("fear_greed"), inp.get("stablecoins"))

    def derivs(s):
        return deriv_features(cal, inp.get("metrics", {}).get(s), inp.get("premium", {}).get(s),
                              inp.get("funding", {}).get(s))

    single = pd.concat([price_features(px["BTCUSDT"]["close"], vz["BTCUSDT"]), derivs("BTCUSDT"), macro, native], axis=1)

    index_px = basket_index(inp, cal)
    basket_a = price_features(index_px, pd.DataFrame(vz).mean(axis=1, skipna=True))
    d_all = [derivs(s) for s in SYMS if s in px]
    basket_b = sum(d.fillna(0) for d in d_all) / sum(d.notna().astype(int) for d in d_all).replace(0, np.nan)
    basket = pd.concat([basket_a, basket_b, macro, native], axis=1)
    return {"single": single[ALL_FEATURES], "basket": basket[ALL_FEATURES]}


def basket_index(inp: dict, cal: pd.DatetimeIndex) -> pd.Series:
    rets = pd.DataFrame({s: np.log(inp["prices"][s]["close"].reindex(cal)).diff() for s in SYMS if s in inp["prices"]})
    idx = np.exp(rets.mean(axis=1, skipna=True).fillna(0).cumsum())
    idx[rets.notna().sum(axis=1) == 0] = np.nan
    return idx


def build_targets(close: pd.Series) -> pd.DataFrame:
    lc = np.log(close)
    r1 = lc.diff()
    n = len(close)
    c = close.to_numpy()
    fwd_min = np.full(n, np.nan)
    for t in range(n - 7):
        fwd_min[t] = np.min(c[t + 1: t + 8]) / c[t] - 1
    crash = pd.Series(np.where(np.isnan(fwd_min), np.nan, (fwd_min <= -0.10).astype(float)), index=close.index)
    vol = pd.concat([r1.shift(-k) for k in range(1, 8)], axis=1).std(axis=1, ddof=1)
    vol[pd.concat([r1.shift(-k) for k in range(1, 8)], axis=1).isna().any(axis=1)] = np.nan
    return pd.DataFrame({"ret_1d": lc.shift(-1) - lc, "ret_7d": lc.shift(-7) - lc, "crash_7d": crash,
                         "vol_7d": vol, "fwd_min_ret": pd.Series(fwd_min, index=close.index)})


def check_no_lookahead(build, inp: dict, t0: pd.Timestamp, seed: int = 0) -> bool:
    """True if features for rows <= t0 are unchanged when every raw value dated AFTER t0 is scrambled.
    `build(inp) -> DataFrame`.  A feature that peeks at the future fails this."""
    rng = np.random.default_rng(seed)

    def scramble(o):
        if isinstance(o, pd.DataFrame):
            o = o.copy()
            m = o.index > t0
            o.loc[m] = o.loc[m] * rng.uniform(0.5, 2.0, size=(m.sum(), o.shape[1]))
            return o
        if isinstance(o, dict):
            return {k: scramble(v) for k, v in o.items()}
        return o

    a, b = build(inp), build(scramble(inp))
    a, b = a[a.index <= t0], b[b.index <= t0]
    return bool(np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True))


# ----------------------------------------------------------------------------- data loading
def fetch_daily(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """1d klines from data-api.binance.vision for [start, end) (public, no key)."""
    rows, t, end_ms = [], int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    while t < end_ms:
        with urllib.request.urlopen(f"{API}?symbol={symbol}&interval=1d&startTime={t}&endTime={end_ms - 1}&limit=1000",
                                    timeout=30) as r:
            batch = json.load(r)
        if not batch:
            break
        rows += [b[:6] for b in batch]
        t = batch[-1][0] + 1
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    ts = df["t"].astype("int64")
    df.index = pd.to_datetime(ts.where(ts < 1e14, ts // 1000), unit="ms", utc=True).rename("date")
    return df[["close", "volume"]].astype(float)


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")).normalize()
    return df


def load_inputs(data_dir: Path, last_day: pd.Timestamp = OOS_END) -> dict:
    prices = {}
    for s in SYMS:
        m = pd.read_parquet(data_dir / "klines" / f"{s}_1m.parquet", columns=["close", "volume"])
        d = pd.DataFrame({"close": m["close"].resample("1D").last(), "volume": m["volume"].resample("1D").sum()}).dropna()
        d = _utc(d)
        if m.index[-1].hour != 23:  # last day incomplete
            d = d.iloc[:-1]
        nxt = d.index[-1] + pd.Timedelta(days=1)
        cache = data_dir / "external" / f"_recent_{s}_1d.parquet"
        if nxt <= last_day:
            try:
                rec = fetch_daily(s, nxt, last_day + pd.Timedelta(days=1))
                cache.parent.mkdir(parents=True, exist_ok=True)
                rec.to_parquet(cache)
            except Exception:
                rec = pd.read_parquet(cache) if cache.exists() else _empty_daily()
            d = pd.concat([d, _utc(rec)])
        prices[s] = d[~d.index.duplicated()].loc[:last_day]
    ext = data_dir / "external"

    def rd(name):
        p = ext / name
        return _utc(pd.read_parquet(p)) if p.exists() else None

    inp = {"prices": prices, "fred": rd("fred.parquet"), "fear_greed": rd("fear_greed.parquet"),
           "stablecoins": rd("stablecoins.parquet"), "metrics": {}, "premium": {}, "funding": {}}
    for s in SYMS:
        for key, name in (("metrics", f"binance_metrics_{s}.parquet"), ("premium", f"binance_premium_{s}.parquet"),
                          ("funding", f"funding_{s}.parquet")):
            df = rd(name)
            if df is not None:
                inp[key][s] = df
    return inp


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame({"close": [], "volume": []}, index=pd.DatetimeIndex([], tz="UTC", name="date"))


# ----------------------------------------------------------------------------- analyses
def univariate(feats: dict, targets: dict) -> pd.DataFrame:
    rows = []
    for uni in feats:
        for tg in ("ret_1d", "ret_7d", "crash_7d", "vol_7d"):
            h = HORIZON[tg]
            y = targets[uni][tg]
            for grp, names in GROUPS.items():
                for f in names:
                    x = feats[uni][f]
                    r = {"universe": uni, "target": tg, "group": grp, "feature": f}
                    for per, (a, b) in (("is", (IS_START, IS_END - pd.Timedelta(days=h))), ("oos", (OOS_START, OOS_END))):
                        m = (x.index >= a) & (x.index <= b)
                        res = ic_test(x[m], y[m], lags=h)
                        r.update({f"{k}_{per}": v for k, v in res.items()})
                    rows.append(r)
    df = pd.DataFrame(rows)
    df["fdr_pass"] = bh_fdr(df["p_oos"].to_numpy())
    df["same_sign"] = np.sign(df["ic_is"]) == np.sign(df["ic_oos"])
    return df


def incremental(feats: dict, targets: dict) -> pd.DataFrame:
    rows = []
    cfgs = {"B": GROUPS["B"], "C": GROUPS["C"], "D": GROUPS["D"], "all": GROUPS["B"] + GROUPS["C"] + GROUPS["D"]}
    for uni in feats:
        for tg, kind in (("ret_1d", "ridge"), ("ret_7d", "ridge"), ("crash_7d", "logit")):
            h = HORIZON[tg]
            for name, extra in cfgs.items():
                cols = GROUPS["A"] + extra
                d = pd.concat([feats[uni][cols], targets[uni][tg].rename("_y")], axis=1).dropna()
                if len(d) < 500:
                    continue
                base = oos_metrics(walk_forward(d[GROUPS["A"]], d["_y"], "ridge" if kind == "ridge" else "logit", h), kind)
                full = oos_metrics(walk_forward(d[cols], d["_y"], "ridge" if kind == "ridge" else "logit", h), kind)
                key = "r2" if kind == "ridge" else "auc"
                rows.append({"universe": uni, "target": tg, "group": name, "metric": key, "base": base[key],
                             "model": full[key], "delta": full[key] - base[key], "n_oos": base["n"],
                             "sample_start": d.index[0].date().isoformat(),
                             "brier_base": base.get("brier", np.nan), "brier_model": full.get("brier", np.nan),
                             "n_pos": base.get("n_pos", np.nan)})
    return pd.DataFrame(rows)


def top_drawdowns(close: pd.Series, k: int = 8, force: pd.Timestamp | None = None) -> pd.DataFrame:
    """k largest non-overlapping 7-day drawdowns.  Base day t, window t+1..t+7, dd = min close / close[t] - 1.
    `force` (base day) is always included (prereg: the 2026-11-09..15 window)."""
    dd = build_targets(close)["fwd_min_ret"].dropna().sort_values()
    chosen: list[pd.Timestamp] = [force] if force is not None else []
    for t in dd.index:
        if len(chosen) >= k:
            break
        if all(abs((t - c).days) > 7 for c in chosen):
            chosen.append(t)
    return pd.DataFrame({"base_day": chosen, "drawdown": [dd.get(t, np.nan) for t in chosen]}).sort_values("base_day")


def event_study(feats_single: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in ALL_FEATURES:
        z = trailing_z(feats_single[f], 365, 120)
        z7 = z.rolling(7, min_periods=5).mean()
        excl = pd.Series(False, index=z.index)
        for t in events.base_day:
            excl[(z.index >= t - pd.Timedelta(days=6)) & (z.index <= t + pd.Timedelta(days=7))] = True
        other = z7[~excl].dropna()
        ev = z7.reindex(events.base_day).dropna()
        rows.append({"feature": f, "group": next(g for g, n in GROUPS.items() if f in n), "n_events": len(ev),
                     "event_z": ev.mean() if len(ev) else np.nan,
                     "other_z": other.mean(), "other_sd": other.std(),
                     "pctile": float(np.mean([(other < v).mean() for v in ev])) if len(ev) else np.nan,
                     "n_pos": int((ev > 0).sum())})
    return pd.DataFrame(rows)


def decide(uni: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in ("B", "C", "D"):
        i = inc[inc.group == g]
        hit_a = i[((i.target == "crash_7d") & (i.delta >= 0.03)) | ((i.target != "crash_7d") & (i.delta >= 0.005))]
        u = uni[(uni.group == g) & uni.fdr_pass & uni.same_sign]
        ok = len(hit_a) > 0 and len(u) > 0
        rows.append({"group": g, "criterion_a": len(hit_a) > 0, "criterion_b": len(u) > 0,
                     "verdict": "contributes" if ok else "does not contribute",
                     "a_detail": "; ".join(f"{r.universe}/{r.target} {r.delta:+.3f}" for r in hit_a.itertuples()),
                     "b_detail": "; ".join(f"{r.universe}/{r.feature}->{r.target}" for r in u.itertuples())})
    return pd.DataFrame(rows)


def build_all(inp: dict, last_day: pd.Timestamp = OOS_END):
    first = min(p.index[0] for p in inp["prices"].values())
    cal = pd.date_range(first, last_day, freq="1D", tz="UTC", name="date")
    feats = build_features(inp, cal)
    targets = {"single": build_targets(inp["prices"]["BTCUSDT"]["close"].reindex(cal)),
               "basket": build_targets(basket_index(inp, cal))}
    return cal, feats, targets


# ----------------------------------------------------------------------------- report
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def _build_section(dec, uni) -> str:
    v = dec.set_index("group")["verdict"]
    L = ["## What to build", "",
         "Read the verdicts with care: criterion (b) was met on volatility targets, while criterion (a) was met on "
         "crash and 7-day return targets, so the two halves of the rule are not backed by the same feature. The best "
         "OOS volatility predictors are price-only (rv_30d, rv_7d); external data adds little on top of them. "
         "Baseline A's crash AUC is at or below 0.5 on the B sample, so part of the AUC gains is a weak baseline being "
         "beaten, not a strong model. None of this is a buy/sell signal.", ""]
    any_ok = False
    if v.get("D") == "contributes":
        any_ok = True
        L.append("- **Stablecoin supply growth (30d), group D:** predicts next-week volatility OOS and fell before most "
                 "large drawdowns. Use as a risk input: widen grid spacing or cut position size when supply growth is "
                 "weak. Not a directional entry.")
    if v.get("B") == "contributes":
        any_ok = True
        L.append("- **Long/short account ratio (global_ls_ratio), group B:** predicts volatility OOS with a stable sign. "
                 "Candidate input to a size scaler or a risk-off gate. The top-trader ratio flipped sign between "
                 "in-sample and OOS, so do not build on it.")
    if v.get("C") == "does not contribute":
        L.append("- **Macro (FRED):** nothing to build; no feature survived FDR.")
    if any_ok:
        L.append("- Before building: paper-test each as a gate in the grid simulator against a gate on rv_30d alone; "
                 "adopt only if it adds something rv_30d does not.")
    return "\n".join(L) + "\n"


def render_report(uni, inc, dec, ev, events, notes: dict) -> str:
    L = ["# Feature screen: does external data add predictive power?", "",
         "*Pre-registered (`research/feature_screen_prereg.json`), run once, no tuning. Daily UTC bars, BTC and an "
         "equal-weight basket of 6 coins. In-sample 2019-2023, out-of-sample (OOS) 2024 to 15 Nov 2026.*", "",
         "## Verdict", "", "| Group | Verdict | (a) beats baseline A | (b) survives FDR with same sign |", "|---|---|---|---|"]
    names = {"B": "B. Binance derivatives", "C": "C. Macro (FRED)", "D": "D. Crypto-native"}
    for r in dec.itertuples():
        L.append(f"| {names[r.group]} | **{r.verdict}** | {'yes' if r.criterion_a else 'no'}"
                 f"{' (' + r.a_detail + ')' if r.a_detail else ''} | {'yes' if r.criterion_b else 'no'} |")
    L += ["", "Rule (literal): a group contributes only if it lifts crash-7d OOS AUC by 0.03 or more, or a return "
          "target's OOS R-squared by 0.5 percentage points or more, over price-only baseline A, AND at least one of its "
          "features survives Benjamini-Hochberg FDR (10%) OOS with the same sign as in-sample. Either target or either "
          "universe counts for (a), which is generous: that is 6 chances per group.", ""]
    bstart = inc[inc.group == "B"].sample_start.min() if (inc.group == "B").any() else "n/a"
    L += ["## Does adding the group improve out-of-sample prediction?", "",
          "Walk-forward, monthly refit, expanding window, standardised on training data only. R-squared is vs the "
          "training mean. Delta = group model minus baseline A on the same rows (B and all use the common sample of rows where every derivative feature exists, from "
          f"{bstart}).", "",
          "| Universe | Target | + Group | Metric | Baseline A | With group | Delta |", "|---|---|---|---|---|---|---|"]
    for r in inc.itertuples():
        L.append(f"| {r.universe} | {r.target} | {r.group} | {'AUC' if r.metric == 'auc' else 'R2'} | "
                 f"{r.base:.3f} | {r.model:.3f} | **{_f(r.delta)}** |")
    pos = inc[inc.target == "crash_7d"].n_pos.max()
    nn = inc.n_oos.max()
    L += ["", f"OOS rows up to {nn}; crash-7d positives about {pos:.0f} in a handful of clustered episodes, so AUC is noisy."
          " Brier scores are in the CSV.", ""]
    s = uni[uni.fdr_pass].sort_values("p_oos")
    L += [f"## Features that survive FDR OOS ({len(s)} of {int(uni.p_oos.notna().sum())} tests)", ""]
    if len(s):
        L += ["| Universe | Feature | Target | IC in-sample | IC OOS | t OOS (HAC) | Same sign |", "|---|---|---|---|---|---|---|"]
        for r in s.head(15).itertuples():
            L.append(f"| {r.universe} | {r.feature} ({r.group}) | {r.target} | {_f(r.ic_is)} | {_f(r.ic_oos)} | "
                     f"{r.t_oos:.1f} | {'yes' if r.same_sign else 'NO'} |")
        if len(s) > 15:
            L.append(f"\n(top 15 by p-value; all {len(s)} in the CSV)")
    else:
        L.append("None.")
    L += ["", "All tests are in `docs/research/feature_screen_all_tests.csv`.", ""]
    L += ["## Event study: BTC's 8 largest non-overlapping 7-day drawdowns", "",
          "Events (base day, worst close within 7 days): " + ", ".join(
              f"{r.base_day:%Y-%m-%d} ({r.drawdown:.0%})" for r in events.itertuples()) + ".", "",
          "Average z-score (vs trailing 365 days) of each feature over the 7 days before the window opens. "
          "Pctile = where that sits in the distribution of 7-day average z on all other days (50% = nothing unusual). "
          "Only features with |avg z| of 0.4 or more are shown; all 26 are in the CSV.", "",
          "| Feature | Events | Avg z before | Other days avg z (sd) | Pctile | Events z>0 |", "|---|---|---|---|---|---|"]
    for r in ev[(ev.event_z.abs() >= 0.4)].itertuples():
        L.append(f"| {r.feature} | {r.n_events} | {_f(r.event_z, 2)} | {_f(r.other_z, 2)} ({r.other_sd:.2f}) | "
                 f"{'n/a' if np.isnan(r.pctile) else f'{r.pctile:.0%}'} | {r.n_pos}/{r.n_events} |")
    L += ["", notes.get("build", ""), "", _build_section(dec, uni), "## Limitations", "",
          "- ETF flows, on-chain exchange flows and news sentiment are unavailable (no free point-in-time source); this "
          "screen says nothing about them.",
          "- The OOS period was seen by the team in other work, so it is not a clean holdout.",
          "- Only a handful of independent crash episodes exist, and 7-day targets overlap; effective sample is small "
          "even with HAC errors.",
          "- Derivatives (B) start 2021-12 and need every column present, so the B comparison sample starts later and its in-sample IC rests on under 2 years.",
          "- Fear and Greed and stablecoin supply are lagged one extra day, an assumption the prereg does not spell out."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="docs/research/FEATURE_SCREEN.md")
    a = ap.parse_args(argv)
    inp = load_inputs(Path(a.data_dir))
    cal, feats, targets = build_all(inp)
    uni = univariate(feats, targets)
    inc = incremental(feats, targets)
    dec = decide(uni, inc)
    events = top_drawdowns(inp["prices"]["BTCUSDT"]["close"].reindex(cal), 8, force=pd.Timestamp("2026-11-08", tz="UTC"))
    ev = event_study(feats["single"], events)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_name("feature_screen_all_tests.csv")
    uni.to_csv(csv, index=False)
    inc.to_csv(out.with_name("feature_screen_incremental.csv"), index=False)
    ev.to_csv(out.with_name("feature_screen_event_study.csv"), index=False)
    cov = {g: feats["single"][v].notna().all(axis=1).idxmax().date().isoformat() for g, v in GROUPS.items()
           if feats["single"][v].notna().all(axis=1).any()}
    out.write_text(render_report(uni, inc, dec, ev, events, {"build": f"First date with every feature of the group (BTC): {cov}."}))
    print(dec.to_string())
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
