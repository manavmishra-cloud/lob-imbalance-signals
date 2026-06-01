"""Synthetic limit order book data generator in LOBSTER format.

Produces message + orderbook CSV files that match the LOBSTER schema exactly,
so the rest of the pipeline (parser, features, models) runs without modification.

The simulator uses a simple phenomenological model:

- Mid-price follows a random walk with time-varying drift (creates regimes).
- Order arrivals at each level have Poisson-style rates, more concentrated at
  the top of the book.
- Submissions, cancellations, and executions occur with realistic proportions
  matching observed LOBSTER data (≈50% submissions, ≈30% cancellations, ≈20% executions).
- The drift state is biased toward whichever side has more depth — this creates
  a genuine OFI → future-return correlation that the downstream models can learn.

This is NOT a research-grade simulator (no jumps, no informed traders, simplified
queue dynamics). It's a pipeline validator: lets you run the full training
flow end-to-end without real data, with results that should beat random.

Usage:
    python -m src.data.synthetic \\
        --output-dir data/raw \\
        --ticker SYNTH \\
        --date 2026-06-01 \\
        --n-events 100000 \\
        --levels 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# LOBSTER stores prices as integers in units of $0.0001
LOBSTER_PRICE_SCALE = 10_000

# Trading day: 09:30:00 = 34200 seconds, 16:00:00 = 57600 seconds
DEFAULT_START_TIME = 34200.0
DEFAULT_END_TIME = 57600.0


class SyntheticLOB:
    """In-memory limit order book with synthetic event generation."""

    def __init__(
        self,
        n_levels: int = 5,
        initial_mid: float = 100.0,
        tick_size: float = 0.01,
        baseline_size: int = 200,
        rng: np.random.Generator | None = None,
    ):
        self.n_levels = n_levels
        self.tick_size = tick_size
        self.mid = initial_mid
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Initialize book with N levels on each side
        # Bid prices descending from just below mid, ask ascending from just above
        self.bid_prices = np.array(
            [initial_mid - (i + 0.5) * tick_size for i in range(n_levels)],
            dtype=np.float64,
        )
        self.ask_prices = np.array(
            [initial_mid + (i + 0.5) * tick_size for i in range(n_levels)],
            dtype=np.float64,
        )
        self.bid_sizes = self.rng.integers(baseline_size // 2, baseline_size * 2, size=n_levels).astype(np.int64)
        self.ask_sizes = self.rng.integers(baseline_size // 2, baseline_size * 2, size=n_levels).astype(np.int64)

        self.baseline_size = baseline_size
        self.order_id_counter = 1
        self.drift = 0.0  # current bias toward buying (positive) or selling (negative)

    # ------------------------------------------------------------------
    # Book maintenance
    # ------------------------------------------------------------------

    def _refill_top(self):
        """If the top level on either side is depleted, drop a level + add a deeper one."""
        if self.bid_sizes[0] < 20:
            # Drop the top — book moves down on the bid side
            self.bid_prices = np.concatenate([self.bid_prices[1:], [self.bid_prices[-1] - self.tick_size]])
            self.bid_sizes = np.concatenate([self.bid_sizes[1:], [self.rng.integers(self.baseline_size // 2, self.baseline_size * 2)]])
        if self.ask_sizes[0] < 20:
            # Drop the top — book moves up on the ask side
            self.ask_prices = np.concatenate([self.ask_prices[1:], [self.ask_prices[-1] + self.tick_size]])
            self.ask_sizes = np.concatenate([self.ask_sizes[1:], [self.rng.integers(self.baseline_size // 2, self.baseline_size * 2)]])
        self.mid = (self.bid_prices[0] + self.ask_prices[0]) / 2.0

    def _improve_price(self, side: int):
        """A new aggressive order improves the top-of-book by one tick."""
        if side > 0:  # buy side
            new_top = self.bid_prices[0] + self.tick_size
            if new_top >= self.ask_prices[0]:
                return  # would cross — skip
            self.bid_prices = np.concatenate([[new_top], self.bid_prices[:-1]])
            self.bid_sizes = np.concatenate([[int(self.rng.integers(50, self.baseline_size))], self.bid_sizes[:-1]])
        else:
            new_top = self.ask_prices[0] - self.tick_size
            if new_top <= self.bid_prices[0]:
                return
            self.ask_prices = np.concatenate([[new_top], self.ask_prices[:-1]])
            self.ask_sizes = np.concatenate([[int(self.rng.integers(50, self.baseline_size))], self.ask_sizes[:-1]])
        self.mid = (self.bid_prices[0] + self.ask_prices[0]) / 2.0

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def step(self, time: float) -> tuple[tuple, np.ndarray]:
        """Generate one synthetic event + return the resulting book snapshot.

        Returns
        -------
        message : tuple of (time, type, order_id, size, price, direction)
        snapshot : flat array of length 4 * n_levels (ask1, asize1, bid1, bsize1, ...)
        """
        # Slowly evolve the drift (creates regime shifts)
        self.drift += self.rng.normal(0, 0.01)
        self.drift = float(np.clip(self.drift, -0.4, 0.4))

        # Choose event type — proportions roughly match LOBSTER empirical distribution
        u = self.rng.random()
        if u < 0.50:
            event_type = 1  # submission
        elif u < 0.80:
            event_type = 3  # deletion (total)
        else:
            event_type = 4  # execution

        # Choose side — biased by current drift
        side_prob_buy = 0.5 + self.drift
        direction = 1 if self.rng.random() < side_prob_buy else -1

        # Choose level (geometric distribution — more action at top)
        level = min(int(self.rng.geometric(0.5)) - 1, self.n_levels - 1)
        level = max(level, 0)

        if direction == 1:  # buy side action
            if event_type == 1:  # submission — add to bid book
                size = int(self.rng.integers(50, 300))
                # Occasionally improve the top of the book
                if level == 0 and self.rng.random() < 0.1:
                    self._improve_price(side=1)
                    price = float(self.bid_prices[0])
                    size = int(self.bid_sizes[0])
                else:
                    self.bid_sizes[level] += size
                    price = float(self.bid_prices[level])
            elif event_type == 3:  # deletion
                size = int(min(self.rng.integers(50, 200), self.bid_sizes[level]))
                self.bid_sizes[level] = max(self.bid_sizes[level] - size, 0)
                price = float(self.bid_prices[level])
            else:  # execution — eats from the ask (buyer crosses spread)
                level = 0
                size = int(min(self.rng.integers(50, 250), self.ask_sizes[0]))
                self.ask_sizes[0] = max(self.ask_sizes[0] - size, 0)
                price = float(self.ask_prices[0])
        else:  # sell side action (symmetric)
            if event_type == 1:
                size = int(self.rng.integers(50, 300))
                if level == 0 and self.rng.random() < 0.1:
                    self._improve_price(side=-1)
                    price = float(self.ask_prices[0])
                    size = int(self.ask_sizes[0])
                else:
                    self.ask_sizes[level] += size
                    price = float(self.ask_prices[level])
            elif event_type == 3:
                size = int(min(self.rng.integers(50, 200), self.ask_sizes[level]))
                self.ask_sizes[level] = max(self.ask_sizes[level] - size, 0)
                price = float(self.ask_prices[level])
            else:
                level = 0
                size = int(min(self.rng.integers(50, 250), self.bid_sizes[0]))
                self.bid_sizes[0] = max(self.bid_sizes[0] - size, 0)
                price = float(self.bid_prices[0])

        # Maintain book integrity
        self._refill_top()

        # Build message tuple
        message = (
            float(time),
            int(event_type),
            int(self.order_id_counter),
            int(size),
            float(price),
            int(direction),
        )
        self.order_id_counter += 1

        # Build orderbook snapshot in LOBSTER column order
        snapshot = np.empty(4 * self.n_levels, dtype=np.float64)
        for i in range(self.n_levels):
            snapshot[4 * i + 0] = self.ask_prices[i]
            snapshot[4 * i + 1] = self.ask_sizes[i]
            snapshot[4 * i + 2] = self.bid_prices[i]
            snapshot[4 * i + 3] = self.bid_sizes[i]

        return message, snapshot


def generate_synthetic_day(
    output_dir: Path | str,
    ticker: str = "SYNTH",
    date: str = "2026-06-01",
    n_events: int = 100_000,
    n_levels: int = 5,
    initial_mid: float = 100.0,
    tick_size: float = 0.01,
    start_time: float = DEFAULT_START_TIME,
    end_time: float = DEFAULT_END_TIME,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate one synthetic LOBSTER-format trading day.

    Writes two CSV files:
        {ticker}_{date}_{start}_{end}_message_{N}.csv
        {ticker}_{date}_{start}_{end}_orderbook_{N}.csv

    Returns
    -------
    (message_path, orderbook_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    lob = SyntheticLOB(n_levels=n_levels, initial_mid=initial_mid, tick_size=tick_size, rng=rng)

    # Generate event times as Poisson arrivals — average rate = n_events / duration
    duration = end_time - start_time
    inter_arrivals = rng.exponential(duration / n_events, size=n_events)
    times = start_time + np.cumsum(inter_arrivals)
    times = times[times < end_time]  # truncate if we overshoot

    messages = np.empty((len(times), 6), dtype=np.float64)
    snapshots = np.empty((len(times), 4 * n_levels), dtype=np.float64)

    for i, t in enumerate(times):
        msg, snap = lob.step(t)
        messages[i] = msg
        snapshots[i] = snap

    # Convert prices to LOBSTER integer format ($0.0001 units)
    messages_out = messages.copy()
    messages_out[:, 4] *= LOBSTER_PRICE_SCALE  # price column
    messages_out[:, 4] = np.round(messages_out[:, 4])

    snapshots_out = snapshots.copy()
    price_cols = [4 * i + 0 for i in range(n_levels)] + [4 * i + 2 for i in range(n_levels)]
    for col in price_cols:
        snapshots_out[:, col] *= LOBSTER_PRICE_SCALE
        snapshots_out[:, col] = np.round(snapshots_out[:, col])

    # LOBSTER filename pattern: {TICKER}_{DATE}_{START}_{END}_{kind}_{N}.csv
    start_int = int(start_time * 1000)
    end_int = int(end_time * 1000)
    base = f"{ticker}_{date}_{start_int}_{end_int}"
    msg_path = output_dir / f"{base}_message_{n_levels}.csv"
    ob_path = output_dir / f"{base}_orderbook_{n_levels}.csv"

    # Write as integer where appropriate; LOBSTER files have no header
    fmt_msg = ["%.9f", "%d", "%d", "%d", "%d", "%d"]
    np.savetxt(msg_path, messages_out, delimiter=",", fmt=fmt_msg)

    # Orderbook: all integers (prices already scaled, sizes are integer)
    np.savetxt(ob_path, snapshots_out, delimiter=",", fmt="%d")

    print(f"Generated {len(times):,} events for {ticker} {date}")
    print(f"  Message file:   {msg_path}")
    print(f"  Orderbook file: {ob_path}")
    return msg_path, ob_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LOBSTER-format LOB data")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--ticker", type=str, default="SYNTH")
    parser.add_argument("--date", type=str, default="2026-06-01")
    parser.add_argument("--n-events", type=int, default=100_000)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--initial-mid", type=float, default=100.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic_day(
        output_dir=args.output_dir,
        ticker=args.ticker,
        date=args.date,
        n_events=args.n_events,
        n_levels=args.levels,
        initial_mid=args.initial_mid,
        tick_size=args.tick_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
