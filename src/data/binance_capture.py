"""Live Binance order book capture for limit order book research.

Subscribes to Binance's partial book depth stream (5 or 20 levels at 100 ms
resolution) and writes a LOBSTER-compatible pair of CSV files:

- ``{SYMBOL}_{DATE}_{ts}_orderbook_{N}.csv`` — orderbook snapshots
- ``{SYMBOL}_{DATE}_{ts}_message_{N}.csv``   — synthetic event log

Each snapshot from Binance becomes one "event" in the message file, with
type 1 (submission) by default. Prices are scaled to LOBSTER's $0.0001
integer convention so the rest of the pipeline (parser, OFI features,
training) runs without modification.

Binance public WebSocket endpoint (no auth required):
  wss://stream.binance.com:9443/ws/{symbol}@depth{N}@100ms

This is a minimal capture utility. For production-grade microstructure
research, prefer the full diff stream (``@depth``) combined with an
initial REST snapshot to maintain a complete local book.

Usage:
    python -m src.data.binance_capture --symbol btcusdt --levels 5 \\
        --duration-minutes 10 --output-dir data/raw

After capture, run training as usual:
    python -m src.training.train_baselines --ticker BTC --target-horizon 5
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import websocket

LOBSTER_PRICE_SCALE = 10_000  # LOBSTER stores prices as int in units of $0.0001


class BinanceDepthCapture:
    def __init__(
        self,
        symbol: str = "btcusdt",
        levels: int = 5,
        duration_seconds: float = 600.0,
        output_dir: Path | str = "data/raw",
    ):
        self.symbol = symbol.lower()
        self.levels = levels
        self.duration_seconds = duration_seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: list[list[float]] = []
        self.start_time: Optional[float] = None
        self._ws: Optional[websocket.WebSocketApp] = None
        self._connected_at: Optional[float] = None

    # ------------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------------
    def _on_open(self, ws):
        self._connected_at = time.time()
        print(f"[binance-capture] connected to {self.symbol}@depth{self.levels}@100ms")

    def _on_message(self, ws, message):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time

        try:
            data = json.loads(message)
            bids = data.get("bids", [])
            asks = data.get("asks", [])
        except json.JSONDecodeError:
            return

        if len(bids) < self.levels or len(asks) < self.levels:
            return

        # LOBSTER column order: ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...
        snapshot = []
        for lvl in range(self.levels):
            ask_p = float(asks[lvl][0])
            ask_s = float(asks[lvl][1])
            bid_p = float(bids[lvl][0])
            bid_s = float(bids[lvl][1])
            snapshot.extend([ask_p, ask_s, bid_p, bid_s])

        self.snapshots.append([elapsed] + snapshot)

        if elapsed >= self.duration_seconds:
            ws.close()

        # Light progress every ~60s
        if len(self.snapshots) % 600 == 0:
            print(f"[binance-capture] {len(self.snapshots):,} snapshots in {elapsed:.0f}s")

    def _on_error(self, ws, error):
        print(f"[binance-capture] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[binance-capture] connection closed after {len(self.snapshots):,} snapshots")

    # ------------------------------------------------------------------
    # Capture lifecycle
    # ------------------------------------------------------------------
    def capture(self) -> tuple[Path, Path]:
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth{self.levels}@100ms"
        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever()

        if not self.snapshots:
            raise RuntimeError("No snapshots captured. Connection may have failed.")

        return self._write_files()

    # ------------------------------------------------------------------
    # File writing (LOBSTER-compatible)
    # ------------------------------------------------------------------
    def _write_files(self) -> tuple[Path, Path]:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts_start = 0
        ts_end = int(self.duration_seconds * 1000)

        # Use UPPER-cased symbol as ticker for the LOBSTER-style filename.
        ticker = self.symbol.upper()
        base = f"{ticker}_{date}_{ts_start}_{ts_end}"
        ob_path = self.output_dir / f"{base}_orderbook_{self.levels}.csv"
        msg_path = self.output_dir / f"{base}_message_{self.levels}.csv"

        arr = np.array(self.snapshots, dtype=np.float64)
        times = arr[:, 0]
        book = arr[:, 1:]  # 4*levels columns

        # Scale prices to LOBSTER integer convention ($0.0001 units)
        price_cols = [4 * i + 0 for i in range(self.levels)] + [4 * i + 2 for i in range(self.levels)]
        for col in price_cols:
            book[:, col] = np.round(book[:, col] * LOBSTER_PRICE_SCALE)
        # Sizes round to int (Binance sizes are float; LOBSTER expects int)
        size_cols = [4 * i + 1 for i in range(self.levels)] + [4 * i + 3 for i in range(self.levels)]
        for col in size_cols:
            book[:, col] = np.round(book[:, col] * 1e6)  # scale 6 decimals for BTC (or similar tokens)

        # Write orderbook
        np.savetxt(ob_path, book, delimiter=",", fmt="%d")

        # Write a minimal synthetic message file (every event = type 1, direction 0, size 0)
        # We don't have true messages from the snapshot stream — this keeps the parser happy.
        n = len(times)
        message = np.zeros((n, 6), dtype=np.float64)
        message[:, 0] = times  # time (seconds from capture start)
        message[:, 1] = 1      # type: submission (placeholder)
        message[:, 2] = np.arange(n)  # order_id (unique sequence)
        message[:, 3] = 100    # size placeholder
        # Use level-1 ask price as the event price
        message[:, 4] = book[:, 0]  # already scaled
        message[:, 5] = 1      # direction: buy (placeholder)

        fmt_msg = ["%.6f", "%d", "%d", "%d", "%d", "%d"]
        np.savetxt(msg_path, message, delimiter=",", fmt=fmt_msg)

        print(f"\n[binance-capture] wrote {n:,} snapshots:")
        print(f"  {ob_path}")
        print(f"  {msg_path}")
        return msg_path, ob_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt", help="Binance symbol (lowercase)")
    parser.add_argument("--levels", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--duration-minutes", type=float, default=10.0)
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    capture = BinanceDepthCapture(
        symbol=args.symbol,
        levels=args.levels,
        duration_seconds=args.duration_minutes * 60.0,
        output_dir=args.output_dir,
    )
    capture.capture()


if __name__ == "__main__":
    main()
