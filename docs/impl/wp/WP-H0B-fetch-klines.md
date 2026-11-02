# WP-H0B Historical 1m klines downloader
Model: haiku · Depends: none · Worktree: no (only touch `scripts/fetch_klines.py`, `tests/test_fetch_klines.py`, `data/klines/`) · Budget: 40k · Commit: NO

## Goal
`scripts/fetch_klines.py` downloads Binance **spot** 1m monthly klines from `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip`. It verifies each zip against its `.CHECKSUM` file (sha256) and writes one parquet per symbol: `data/klines/{SYMBOL}_1m.parquet`.

## Spec
- CLI: `--symbols ETHUSDT BTCUSDT --start 2019-01 --end 2026-10 --out data/klines`
- Use stdlib `urllib` (it honours HTTPS_PROXY) or `requests`. Retry 3 times with backoff. Skip months that return 404, and log them.
- **Timestamp gotcha:** Binance spot files from 2025-01 onward use **microsecond** timestamps; earlier files use milliseconds. Normalise everything to UTC `datetime64[ns]` (detect µs by magnitude > 1e14).
- CSVs have no header. Columns: open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore. Keep `open_time` (index, UTC), `open`, `high`, `low`, `close`, `volume` (float64), `trades` (int64).
- Sort, drop duplicate timestamps, and report the gap count (missing minutes) per symbol.
- Delete each zip after it parses. Resume-safe: cache parsed months as `data/klines/_parts/{SYMBOL}-{YYYY-MM}.parquet` and skip ones already present.
- Use the scratchpad venv (`/tmp/claude-0/-home-user-VolatilityViking/01f0721a-1d36-5a2d-8343-532333f56486/scratchpad/venv/bin/python`); pip install `pyarrow` there if it's missing. Don't touch `.venv` (another agent is building it).

## Test (write it first)
`tests/test_fetch_klines.py`: the timestamp normaliser handles both a ms and a µs sample row; the checksum verifier rejects a tampered byte string. No network in tests.

## Run
Run the real download for ETHUSDT and BTCUSDT, 2019-01 to 2026-10.

## Done when
Both parquet files exist. Report per symbol: rows, first and last timestamp, gap count, file size.

## Reply (≤120 words)
status · rows/range/gaps/size per symbol · test result · months skipped
