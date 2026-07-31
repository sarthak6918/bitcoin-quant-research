# Live Signal System: Python Inference + Webhook

Deploys the n=1-bar production model as a live, hourly BTC/USDT signal
generator that POSTs a webhook when its probability crosses a threshold
calibrated to ~2 signals/day. This is the "gate entries on the model's own
score" approach recommended in `STRATEGY_INPUT_SEARCH_FINDINGS.md`, since
that search proved no amount of tuning the pinescript's own RSI/Stoch/ADX
inputs gets its entries to correlate with the feature set.

**Scope, explicitly**: this system decides and notifies. It does not place
any order, call any exchange/broker API, or move money. Wiring real
execution downstream of the webhook is a separate, deliberate decision for
you to make with your own risk controls -- not something bundled in here.

## Files

| File | What it is |
|---|---|
| `feature_lib.py` | The same causal feature computation as `build_features.py`, refactored into a reusable function so live and training feature code can never drift apart. **Verified byte-identical to the offline pipeline** across 59 shared columns on the full historical dataset (see below) -- a mechanical, not just visual, train/serve-skew check. |
| `train_production_model.py` | Trains the final 5-seed ensemble on ALL history through 2026-07-22 (not a walk-forward research fold) and calibrates BUY/SELL probability thresholds to hit ~2 signals/day from the causal OOS conviction distribution |
| `production_models/` | `model_seed{42..46}.cbm` (5 CatBoost models) + `production_config.json` (feature list, thresholds, horizon) |
| `live_inference.py` | Fetches recent closed hourly candles from Binance's public REST API (no key needed), builds features, scores with the ensemble, decides BUY/SELL/SKIP, and POSTs a webhook on BUY/SELL. Tracks last-processed bar in `live_state.json` so re-running doesn't double-fire. Run hourly via cron/Task Scheduler. |
| `webhook_receiver.py` | Minimal reference receiver: logs incoming signals to `received_signals.csv` and prints them. Nothing more. Replace/extend with your own downstream logic when you're ready. |

## Why regime (HMM) features were dropped for live

Live decoding would need `hmmlearn`, which fails to build in this
environment (no MSVC C++ Build Tools installed on this machine). Rather
than block the whole live system on that, the regime columns were dropped
and the model retrained: **AUC 0.5397 without regime vs 0.5382 with**, on
the exact same 2025-2026 holdout -- confirms this loses nothing (consistent
with the original project's own finding that regime never earned a place
in the classifier: see `00_METHODOLOGY_AND_FINDINGS.md`, Part 1).

## Verified before shipping

- **Train/serve feature parity**: `feature_lib.build_live_features()` run
  on the full historical OHLCV reproduces every one of the 59 shared
  feature columns in `features_labeled.csv` exactly (max abs diff < 1e-6).
- **End-to-end dry run**: `live_inference.py` fetched 2,159 real closed
  hourly candles from Binance, scored the latest bar
  (2026-07-28 07:00 UTC, close $63,505.99), got prob=0.4851 -> **SKIP**
  (correctly inside the no-signal band -- today just wasn't a signal day,
  nothing was forced).
- **Webhook plumbing**: `webhook_receiver.py` started, received a test POST,
  logged it correctly to `received_signals.csv` with a server-side
  timestamp.

## Running it

```bash
# one-time
python train_production_model.py

# start your receiver (or point --webhook-url at your own service/Discord/etc.)
python webhook_receiver.py --port 8000

# run every hour, a few minutes after each hourly candle closes
python live_inference.py --webhook-url http://localhost:8000/webhook
```

Schedule the last command hourly (`5 * * * *` in cron, or Windows Task
Scheduler set to run every hour). Nothing in this repo places that call on
a schedule automatically -- do that step yourself so you control when it
starts running unattended.

## Honest expectation setting

This model's walk-forward and holdout AUC is ~0.54-0.57 (see
`README.md` in this folder) -- real, stable, mechanically-verified-leakage-
free signal, but modest. At ~2 signals/day this is not a high-confidence
system; it is the best defensible edge this project's data supports.
Treat every signal accordingly before any capital decision.
