"""
Minimal reference webhook receiver for live_inference.py's signal POSTs.

This ONLY logs incoming signals to a CSV and prints them -- it does not
place orders, call any broker/exchange API, or move money. Wire your own
execution logic downstream deliberately, with your own risk controls,
after reviewing signals here; that decision is yours to make explicitly,
not something to bolt on silently.

Run:  python webhook_receiver.py --port 8000
Test: curl -X POST localhost:8000/webhook -H "Content-Type: application/json" \
        -d '{"symbol":"BTCUSDT","timestamp":"2026-07-28T12:00:00","action":"BUY","probability":0.71,"close_price":65000,"horizon_bars":1}'
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG_FILE = Path("received_signals.csv")
FIELDS = ["received_at", "symbol", "timestamp", "action", "probability", "close_price", "horizon_bars"]


def append_log(row):
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/webhook":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"invalid json")
            return

        row = {k: payload.get(k, "") for k in FIELDS if k != "received_at"}
        row["received_at"] = datetime.now(timezone.utc).isoformat()
        append_log(row)
        print(f"[{row['received_at']}] {row['action']} {row['symbol']} "
              f"prob={row['probability']} close={row['close_price']} bar={row['timestamp']}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "logged"}).encode())

    def log_message(self, format, *args):
        pass  # keep stdout clean, our own prints above are enough


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"listening on :{args.port}/webhook -- logging to {LOG_FILE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
