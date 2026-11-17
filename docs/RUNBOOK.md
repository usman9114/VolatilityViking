# VolatilityViking runbook (operator)

Plain-language guide for running the daily bot on the VM. Read it end to end once before the first start.

## 1. The VM
- Put the VM in a region where Binance USD-M futures are legal for you. Check the rules before you rent it.
- Install Docker and Docker Compose, and run NTP (`chrony`). Clock drift breaks signed requests.
- Firewall: allow outbound HTTPS only. Open no inbound ports.
- Keep the checkout at `/opt/volatilityviking`, or edit `WorkingDirectory=` in `deploy/systemd/*.service`.
- The container runs as uid 1000. Let it write to the output folders:
  `sudo chown -R 1000:1000 state reports data`

## 2. API key
- Use a dedicated sub-account, not your main account.
- **Withdrawals disabled.** Futures enabled. Spot trading enabled only if the config uses spot.
- IP allowlist with the VM's public IP only.
- Start with testnet keys from testnet.binancefuture.com. Real keys come only after stage 2.

## 3. `.env` (never commit it)
Create `.env` in the checkout and run `chmod 600 .env`. Variables:
- `VV_BINANCE_KEY`, `VV_BINANCE_SECRET`: the exchange key pair.
- `VV_BINANCE_TEST_VENUE`: only for `--mode testnet`. Set it to `demo` (the default) for keys made in Binance Demo Trading (demo-fapi.binance.com), or `testnet` for older keys made on testnet.binancefuture.com. As of Nov 2026 the testnet site redirects to Demo Trading, so new keys are Demo keys. A key only works on the venue it was created on.
- `VV_TELEGRAM_TOKEN`, `VV_TELEGRAM_CHAT`: alert bot and chat.
- `VV_HEARTBEAT_URL`: a dead-man's-switch URL, pinged after every run.
- `VV_LIVE_CONFIRM=yes`: only for live. Leave it out otherwise. Without it, live mode refuses to run.

Copy `config/runner.example.yaml` to `config/runner.yaml` and set `mode:` (dry, testnet or live).

## 4. Stages and exit criteria
Each stage must meet its exit criteria before the next one starts. The criteria come from `docs/AI_SIGNAL_PLAN.md` §5, plus the stage lengths set here.

1. **Dry, 1 week.** Runs every day with no crash. Read every daily report. Exit: 7 consecutive runs exit 0, and the report numbers match a hand check of one day.
2. **Testnet, 2 weeks, with the drills.** Exit: drills 1 to 5 pass on the VM (drills 3 and 5 are VM-only), with no unexplained difference between testnet positions and the bot's state.
3. **Live shadow.** Real data and keys, no capital moved: signals run in shadow, and shadow P&L is compared with the simulator. Exit: tracking error is inside the band in `docs/AI_SIGNAL_PLAN.md` §5. Live minus shadow daily P&L must have a mean absolute value of 10 bps of equity or less, and the cumulative gap over 4 weeks must be 1% of equity or less.
4. **Tiny live, only with the owner's written approval.** Exit to scale up: zero ops incidents for 4 weeks, and live vs shadow tracking inside that band. A breach halts the ramp-up. Drills must pass before this stage starts.

## 5. Schedule
- 23:40 UTC: `vv-refresh` pulls the external data (incremental).
- 00:05 UTC: `vv-runner` trades the day.

Install:
`sudo cp deploy/systemd/vv-* /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now vv-refresh.timer vv-runner.timer`

Both timers use `Persistent=true`, so a run missed during a reboot happens at the next boot.

## 6. Daily check (about 5 minutes)
1. Open `reports/YYYY-MM-DD.md` for the day.
2. Read the Telegram alerts. No alert means no action.
3. Heartbeat missing? Treat the run as failed and go to section 7.

## 7. Exit codes and halts
| Code | Meaning | What to do |
|---|---|---|
| 0 | OK | Nothing |
| 1 | Error (see the alert) | Read the log, fix the cause, wait for the next run |
| 2 | Data gap: trading skipped | Usually network or a late exchange file. Check the next run |
| 3 | Halted: a persisted halt is set | Read the reason (below). Do not resume blindly |
| 4 | Reconcile or startup refused | Positions do not match the bot's state. Reconcile by hand, then resume |
| 5 | Live refused | `VV_LIVE_CONFIRM=yes` is missing. Intended, unless you meant to go live |

**Reading a halt:** the alert and the daily report both show the stored reason. Check the report and the exchange account first.

**Resuming:** `docker compose run --rm runner resume --reason "what you checked and why it is safe"`. Never resume without understanding the cause. The reason is written to the log.

## 8. Updating
1. `git pull`
2. `docker compose build`
3. Run the tests on a full checkout with `make test`. The image does not contain the `tests/` folder or pytest.
4. Run drills 2 and 4 again (section 9).
5. Note the change in your own change log.

## 9. Ops drills
Scripts are in `deploy/drills/`. Each prints PASS, FAIL or SKIP.
- Drills 1, 2 and 4 run in dry mode. They use the checkout's Python environment (`make bootstrap`), not the container, because the seed helper runs on the host.
- Drills 3 and 5 are VM-only. Run them on the VM with `VV_DRILL_VM=1`, in testnet mode.


## Creating a test API key

Use one of the two options. Both use play money.

**Option A: Futures testnet (testnet.binancefuture.com). Legacy: the site now redirects to Demo Trading, so use Option B for new keys.**
1. Open https://testnet.binancefuture.com and register or log in. This is a separate account from your real Binance login.
2. On the trading page, scroll below the chart to the **API Key** tab. Your key and secret are shown there.
3. Copy both straight into the VM's `.env`. The secret is shown only once.
4. Set `VV_BINANCE_TEST_VENUE=testnet`.

**Option B: Binance Demo Trading (demo-fapi.binance.com)**
1. Log in to binance.com and switch to **Demo Trading**.
2. Open **API Management** inside Demo Trading and create a key. Choose system-generated (HMAC).
3. Enable futures trading only. Leave withdrawals off.
4. Copy the key and secret into `.env`. `VV_BINANCE_TEST_VENUE` defaults to `demo`.

Binance changes these menus from time to time. If a button has moved, check https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info.

**Never paste keys into chats, tickets or git.** They belong only in `.env` (chmod 600) on the VM.

**Smoke test** (run on the VM after `.env` is set, before the first testnet run):
```bash
set -a; . ./.env; set +a
docker compose run --rm -e PYTHONPATH=/app --entrypoint python runner deploy/testnet_smoke.py --symbol BTC/USDT:USDT
# or, from a checkout with the dev venv:  PYTHONPATH=. .venv/bin/python deploy/testnet_smoke.py
```
It should end with `SMOKE PASS`. It never prints keys.
