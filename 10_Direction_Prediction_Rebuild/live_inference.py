"""
Live signal generator: fetches recent BTC/USDT hourly OHLCV from Binance's
public REST API (no key required), builds the exact same causal feature
set used in training (feature_lib.py -- verified byte-identical to the
offline pipeline), scores it with the 5-seed production CatBoost ensemble,
and POSTs a webhook when the averaged probability crosses one of the
calibrated thresholds (~2 signals/day combined).

Run this once per hour, on the hour, a few minutes after each hourly candle
closes (e.g. via cron `5 * * * *` or Windows Task Scheduler). It only acts
on CLOSED candles -- the in-progress candle is always dropped before
feature computation, so there is no intra-bar lookahead.

This script only DECIDES and NOTIFIES. It does not place any order or move
any money -- see webhook_receiver.py for a reference logging endpoint, and
wire your own execution/broker logic downstream of the webhook deliberately,
with your own risk controls, at your own decision.

Config:
  WEBHOOK_URL   env var, or --webhook-url flag. Required to actually send.
  If unset, the script still runs and prints what it WOULD have sent.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier

from feature_lib import build_live_features

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOOKBACK_HOURS = 90 * 24  # 90 days -- generous margin over the ~220h warmup actually needed
STATE_FILE = Path("live_state.json")
MODEL_DIR = Path("production_models")


def fetch_recent_klines(hours=LOOKBACK_HOURS):
    limit = 1000
    end_time = None
    all_rows = []
    remaining = hours
    while remaining > 0:
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": min(limit, remaining)}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows = rows + all_rows
        end_time = rows[0][0] - 1
        remaining -= len(rows)
        if len(rows) < params["limit"]:
            break
        time.sleep(0.2)

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qav", "trades", "taker_base", "taker_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume", "close_time"]]
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    now = pd.Timestamp.now("UTC").tz_localize(None)
    df = df[df["close_time"] <= now].reset_index(drop=True)
    return df.drop(columns=["close_time"])


def load_models(model_dir=MODEL_DIR):
    with open(model_dir / "production_config.json") as f:
        config = json.load(f)
    models = []
    for seed in config["seeds"]:
        m = CatBoostClassifier()
        m.load_model(str(model_dir / f"model_seed{seed}.cbm"))
        models.append(m)
    return models, config


def score_latest(models, config, ohlcv_df):
    feat = build_live_features(ohlcv_df)
    latest = feat.iloc[[-1]]
    feature_cols = config["feature_cols"]
    if latest[feature_cols].isna().any(axis=1).item():
        raise RuntimeError(
            "NaN in latest feature row -- insufficient warmup history fetched. "
            "Increase LOOKBACK_HOURS."
        )
    X = latest[feature_cols]
    probs = [m.predict_proba(X)[:, 1][0] for m in models]
    prob = float(np.mean(probs))
    return prob, latest["timestamp"].iloc[0], latest["close"].iloc[0]


def decide_action(prob, config):
    if prob >= config["buy_threshold"]:
        return "BUY"
    if prob <= config["sell_threshold"]:
        return "SELL"
    return "SKIP"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_processed_timestamp": None}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def send_webhook(url, payload):
    if not url:
        print("[no webhook url set -- would have sent]:", json.dumps(payload))
        return
    resp = requests.post(url, json=payload, timeout=10)
    print(f"webhook POST -> {resp.status_code}: {resp.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhook-url", default=os.environ.get("WEBHOOK_URL"))
    ap.add_argument("--force", action="store_true", help="score and print even if this bar was already processed")
    args = ap.parse_args()

    ohlcv = fetch_recent_klines()
    print(f"fetched {len(ohlcv)} closed hourly candles, latest close: {ohlcv['timestamp'].iloc[-1]}")

    models, config = load_models()
    prob, ts, close_price = score_latest(models, config, ohlcv)
    action = decide_action(prob, config)

    state = load_state()
    already_processed = (str(ts) == state.get("last_processed_timestamp"))

    print(f"timestamp={ts}  close={close_price:.2f}  prob={prob:.4f}  action={action}"
          f"{'  [already processed, skipping webhook]' if already_processed and not args.force else ''}")

    if action != "SKIP" and (not already_processed or args.force):
        payload = {
            "symbol": SYMBOL,
            "timestamp": str(ts),
            "action": action,
            "probability": round(prob, 4),
            "close_price": close_price,
            "horizon_bars": config["horizon_bars"],
        }
        send_webhook(args.webhook_url, payload)

    state["last_processed_timestamp"] = str(ts)
    save_state(state)


if __name__ == "__main__":
    main()
