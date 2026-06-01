"""LOBSTER limit order book data parser.

LOBSTER provides two files per stock-day:

1. **Message file** (`*_message_N.csv`): one row per market event.
   Columns: time, type, order_id, size, price, direction

   Event types:
       1 = Submission of a new limit order
       2 = Cancellation (partial deletion of a limit order)
       3 = Deletion (total deletion of a limit order)
       4 = Execution of a visible limit order
       5 = Execution of a hidden limit order
       6 = Cross trade (auction trade)
       7 = Trading halt

   Direction:
       -1 = Sell limit order
       +1 = Buy limit order

2. **Orderbook file** (`*_orderbook_N.csv`): book state after each event.
   Columns interleaved as: ask_price_1, ask_size_1, bid_price_1, bid_size_1,
   ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...
   (N levels deep, where N is 1, 5, 10, 30, or 50)

LOBSTER prices are stored as integers in units of $0.0001 (1/10000 of a dollar).
Times are seconds after midnight on the trading day.

Reference: https://lobsterdata.com/info/DataStructure.php
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOBSTER_PRICE_SCALE = 10_000  # LOBSTER stores prices as int in units of $0.0001

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]

EVENT_TYPE_NAMES = {
    1: "submission",
    2: "cancellation_partial",
    3: "cancellation_total",
    4: "execution_visible",
    5: "execution_hidden",
    6: "cross_trade",
    7: "trading_halt",
}


@dataclass
class LobsterFiles:
    """Resolved paths to a LOBSTER message+orderbook pair."""

    message: Path
    orderbook: Path
    ticker: str
    date: str
    levels: int
    start_time: int
    end_time: int


def find_lobster_pair(
    data_dir: Path | str,
    ticker: str,
    date: Optional[str] = None,
    levels: Optional[int] = None,
) -> LobsterFiles:
    """Locate a matching message + orderbook pair in `data_dir`.

    LOBSTER file naming convention:
        {TICKER}_{YYYY-MM-DD}_{START}_{END}_{message|orderbook}_{N}.csv

    Example:
        AAPL_2012-06-21_34200000_57600000_message_5.csv
        AAPL_2012-06-21_34200000_57600000_orderbook_5.csv

    Parameters
    ----------
    data_dir : Path | str
        Directory containing extracted LOBSTER CSV files.
    ticker : str
        Stock ticker (e.g. "AAPL").
    date : str, optional
        Date in YYYY-MM-DD form. If None, returns first match for ticker.
    levels : int, optional
        Number of book levels (1, 5, 10, 30, 50). If None, returns first match.

    Returns
    -------
    LobsterFiles
        Resolved paths and metadata.
    """
    data_dir = Path(data_dir)
    pattern = f"{ticker}_*_message_*.csv"
    if date is not None:
        pattern = f"{ticker}_{date}_*_message_*.csv"

    candidates = sorted(data_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No LOBSTER message file matching {pattern} in {data_dir}. "
            f"Download samples from https://lobsterdata.com/info/DataSamples.php "
            f"and extract into {data_dir}."
        )

    for msg_path in candidates:
        # Filename: AAPL_2012-06-21_34200000_57600000_message_5.csv
        parts = msg_path.stem.split("_")
        if len(parts) < 6:
            continue
        ticker_p, date_p, start_p, end_p, kind_p, levels_p = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        if kind_p != "message":
            continue
        if levels is not None and int(levels_p) != levels:
            continue

        ob_name = msg_path.name.replace("_message_", "_orderbook_")
        ob_path = msg_path.parent / ob_name
        if not ob_path.exists():
            continue

        return LobsterFiles(
            message=msg_path,
            orderbook=ob_path,
            ticker=ticker_p,
            date=date_p,
            levels=int(levels_p),
            start_time=int(start_p),
            end_time=int(end_p),
        )

    raise FileNotFoundError(
        f"Found {len(candidates)} message files but none paired with an orderbook file "
        f"(or matching the requested levels={levels})."
    )


def load_messages(path: Path | str) -> pd.DataFrame:
    """Load a LOBSTER message file.

    Returns a DataFrame with columns:
        time (float, seconds after midnight)
        type (int, event type code)
        type_name (str, human-readable event type)
        order_id (int)
        size (int, shares)
        price (float, dollars — scaled from LOBSTER's integer cents)
        direction (int, +1=buy, -1=sell)
    """
    df = pd.read_csv(path, header=None, names=MESSAGE_COLUMNS)
    df["price"] = df["price"] / LOBSTER_PRICE_SCALE
    df["type_name"] = df["type"].map(EVENT_TYPE_NAMES)
    return df


def load_orderbook(path: Path | str, levels: int) -> pd.DataFrame:
    """Load a LOBSTER orderbook file.

    Returns a DataFrame with 4*levels columns, prices already scaled to dollars:
        ask_price_1, ask_size_1, bid_price_1, bid_size_1,
        ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...
    """
    columns = []
    for lvl in range(1, levels + 1):
        columns.extend([f"ask_price_{lvl}", f"ask_size_{lvl}", f"bid_price_{lvl}", f"bid_size_{lvl}"])

    df = pd.read_csv(path, header=None, names=columns)

    # Scale all price columns from LOBSTER's integer cents to dollars
    price_cols = [c for c in df.columns if "price" in c]
    df[price_cols] = df[price_cols] / LOBSTER_PRICE_SCALE
    return df


def load_paired(
    files: LobsterFiles,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a message + orderbook pair as aligned DataFrames.

    The orderbook file has one row per event in the message file, in the same order.
    This function aligns them by row index (no time-join needed).

    Returns
    -------
    messages : pd.DataFrame
    orderbook : pd.DataFrame
    """
    messages = load_messages(files.message)
    orderbook = load_orderbook(files.orderbook, levels=files.levels)

    if len(messages) != len(orderbook):
        raise ValueError(
            f"Message file has {len(messages)} rows but orderbook file has "
            f"{len(orderbook)} rows — they should match exactly."
        )

    return messages, orderbook


def compute_mid_price(orderbook: pd.DataFrame) -> pd.Series:
    """Compute the mid-price series from the level-1 book."""
    return (orderbook["ask_price_1"] + orderbook["bid_price_1"]) / 2.0


def compute_microprice(orderbook: pd.DataFrame) -> pd.Series:
    """Compute the size-weighted mid (microprice).

    microprice = (ask_price * bid_size + bid_price * ask_size) / (ask_size + bid_size)

    The microprice is biased toward the side with more competition; it tends to
    predict the next mid-price move better than the simple mid.
    """
    a, ap = orderbook["ask_size_1"], orderbook["ask_price_1"]
    b, bp = orderbook["bid_size_1"], orderbook["bid_price_1"]
    return (ap * b + bp * a) / (a + b)


def compute_spread(orderbook: pd.DataFrame) -> pd.Series:
    """Bid-ask spread at level 1."""
    return orderbook["ask_price_1"] - orderbook["bid_price_1"]
