# WP-R1 External data fetchers (daily)
Model: haiku · Depends: none · Worktree: no (only touch `research/external_fetch.py`, `tests/research/test_external_fetch.py`, `data/external/`) · Budget: 60k · Commit: NO

## Goal
Download the external series listed in `research/feature_screen_prereg.json` into `data/external/` as parquet. Each file must have a UTC daily `DatetimeIndex` named `date`, sorted, no duplicates, and numeric columns **exactly** as named below. Another agent is writing the analysis against these names right now.

| File | Source (all reachable via the env proxy; use `urllib`) | Columns |
|---|---|---|
| `fred.parquet` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>` for DTWEXBGS, DGS10, VIXCLS, NASDAQCOM, WALCL, RRPONTSYD, WTREGEN | one column per FRED id (float; "." means missing → NaN). Raw observation dates, **no lagging here** |
| `fear_greed.parquet` | `https://api.alternative.me/fng/?limit=0&format=json` | `fear_greed` |
| `stablecoins.parquet` | `https://stablecoins.llama.fi/stablecoincharts/all` (sum `totalCirculatingUSD.peggedUSD`) | `stablecoin_supply_usd` |
| `binance_metrics_{SYM}.parquet` for BTCUSDT, ETHUSDT, SOLUSDT | daily zips `https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{YYYY-MM-DD}.zip` from 2021-12-01 to 2026-11-15 (CSV with header; 5-minute rows). Aggregate each day to its **last** row | `sum_open_interest`, `sum_open_interest_value`, `count_toptrader_long_short_ratio`, `sum_toptrader_long_short_ratio`, `count_long_short_ratio`, `sum_taker_long_short_vol_ratio` |
| `binance_premium_{SYM}.parquet` for the same 3 | monthly `https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/{SYM}/1h/{SYM}-1h-{YYYY-MM}.zip` from 2020-01 (skip 404s), plus daily zips for 2026-11-01..15 | `perp_premium` = daily mean of hourly premium closes |
| `funding_{SYM}.parquet` for all 6 coins | reuse `data/futures/` from the carry study if present (check `research/carry_fetch.py` for paths), else monthly `fundingRate` zips | `funding_rate` = daily sum of funding prints |

## Rules
- Downloads are untrusted data: write zips to a fresh dir under `data/external/_raw/` and parse them with pandas only. Delete the zips after parsing.
- Run downloads concurrently (`ThreadPoolExecutor`, max 8 workers), retry 3 times, and skip 404s while recording them.
- Timestamps: Binance files use ms before 2025 and µs from 2025. Normalise by magnitude.
- Use `.venv/bin/python`.

## Tests first
`tests/research/test_external_fetch.py` (no network):
- FRED "." → NaN.
- The metrics day aggregation takes the last row.
- ms/µs normalisation.
- The output index is daily UTC and named `date`.

## Done when
All files exist. Print a table of file → rows, first date, last date, NaN share. Tests and ruff are green.

## Reply (≤120 words)
status · that table · any source that failed or is shorter than expected
