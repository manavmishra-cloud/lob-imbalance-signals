"""Feature engineering for limit order book data.

Implements the classical microstructure features used as inputs to short-horizon
price-direction models:

- **Order Flow Imbalance (OFI)** — Cont, Kukanov, Stoikov (2014)
- **Volume order imbalance** at multiple book depths
- **Bid-ask spread** and **mid-price returns**
- **Microprice deviation** from mid
- **Book depth ratios**
- **Trade flow imbalance** over rolling windows

Each function returns either a Series or DataFrame indexed the same as the input
orderbook DataFrame, so they compose easily.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lobster import compute_mid_price, compute_microprice, compute_spread


# ---------------------------------------------------------------------------
# Order Flow Imbalance (Cont, Kukanov, Stoikov 2014)
# ---------------------------------------------------------------------------

def order_flow_imbalance(
    orderbook: pd.DataFrame,
    level: int = 1,
) -> pd.Series:
    """Compute OFI at a single book level following Cont et al. (2014).

    For each book update at time t (vs. t-1):

        e_t = I(P^b_t >= P^b_{t-1}) * q^b_t                          (bid side)
              - I(P^b_t <= P^b_{t-1}) * q^b_{t-1}
              - I(P^a_t <= P^a_{t-1}) * q^a_t                        (ask side)
              + I(P^a_t >= P^a_{t-1}) * q^a_{t-1}

    Where P^b, P^a are the bid/ask prices and q^b, q^a the sizes at the level.

    Intuition: positive OFI = net buying pressure (bid side strengthens or ask
    side weakens), negative OFI = net selling pressure.
    """
    bp = orderbook[f"bid_price_{level}"].values
    bs = orderbook[f"bid_size_{level}"].values
    ap = orderbook[f"ask_price_{level}"].values
    asz = orderbook[f"ask_size_{level}"].values

    bp_prev, bs_prev = np.roll(bp, 1), np.roll(bs, 1)
    ap_prev, as_prev = np.roll(ap, 1), np.roll(asz, 1)

    # Cast boolean masks to int explicitly — newer numpy disallows unary `-` on bool arrays.
    bid_up = (bp >= bp_prev).astype(np.int64)
    bid_down = (bp <= bp_prev).astype(np.int64)
    ask_up = (ap >= ap_prev).astype(np.int64)
    ask_down = (ap <= ap_prev).astype(np.int64)

    # Bid side contribution
    bid_term = bid_up * bs - bid_down * bs_prev

    # Ask side contribution (negative — ask side weakening = buying pressure)
    ask_term = -ask_down * asz + ask_up * as_prev

    ofi = (bid_term + ask_term).astype(np.float64)
    ofi[0] = 0.0  # First observation has no previous reference
    return pd.Series(ofi, index=orderbook.index, name=f"ofi_L{level}")


def multi_level_ofi(orderbook: pd.DataFrame, max_level: int = 5) -> pd.DataFrame:
    """OFI at each of the first `max_level` book levels.

    Returns a DataFrame with columns ofi_L1, ofi_L2, ..., ofi_LN.
    """
    cols = {}
    for lvl in range(1, max_level + 1):
        if f"bid_price_{lvl}" not in orderbook.columns:
            break
        cols[f"ofi_L{lvl}"] = order_flow_imbalance(orderbook, level=lvl)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Volume / depth based imbalances
# ---------------------------------------------------------------------------

def volume_imbalance(orderbook: pd.DataFrame, level: int = 1) -> pd.Series:
    """Normalized depth imbalance at a single level.

        (bid_size - ask_size) / (bid_size + ask_size)

    Range: -1 (only asks) to +1 (only bids).
    """
    b = orderbook[f"bid_size_{level}"]
    a = orderbook[f"ask_size_{level}"]
    return ((b - a) / (b + a)).rename(f"vol_imb_L{level}")


def cumulative_depth_imbalance(orderbook: pd.DataFrame, max_level: int = 5) -> pd.Series:
    """Total bid depth vs. ask depth across the first `max_level` levels.

    Returns the normalized difference of cumulative volume.
    """
    bid_cols = [f"bid_size_{lvl}" for lvl in range(1, max_level + 1) if f"bid_size_{lvl}" in orderbook.columns]
    ask_cols = [f"ask_size_{lvl}" for lvl in range(1, max_level + 1) if f"ask_size_{lvl}" in orderbook.columns]

    bid_total = orderbook[bid_cols].sum(axis=1)
    ask_total = orderbook[ask_cols].sum(axis=1)
    return ((bid_total - ask_total) / (bid_total + ask_total)).rename("cum_depth_imb")


# ---------------------------------------------------------------------------
# Price-based features
# ---------------------------------------------------------------------------

def mid_price_return(orderbook: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Forward log-return of the mid-price over `horizon` events.

    For prediction targets, use `horizon > 0`; for lagged features use a shift.
    """
    mid = compute_mid_price(orderbook)
    return np.log(mid.shift(-horizon) / mid).rename(f"mid_ret_h{horizon}")


def micro_minus_mid(orderbook: pd.DataFrame) -> pd.Series:
    """Microprice deviation from mid, normalized by spread.

    Positive = microprice above mid (buying pressure).
    """
    mid = compute_mid_price(orderbook)
    micro = compute_microprice(orderbook)
    spread = compute_spread(orderbook)
    # Avoid division by zero when spread is ~0
    return ((micro - mid) / spread.replace(0, np.nan)).rename("micro_dev")


def spread_in_ticks(orderbook: pd.DataFrame, tick_size: float = 0.01) -> pd.Series:
    """Bid-ask spread expressed in ticks."""
    return (compute_spread(orderbook) / tick_size).rename("spread_ticks")


# ---------------------------------------------------------------------------
# Trade flow imbalance (from message file)
# ---------------------------------------------------------------------------

def trade_flow_imbalance(
    messages: pd.DataFrame,
    rolling_seconds: float = 1.0,
) -> pd.Series:
    """Trade flow imbalance over a rolling time window.

    For each row in messages, sums signed executed volume over the trailing
    `rolling_seconds` window. Direction: +1 = buyer-initiated, -1 = seller-initiated.

    Only event types 4 (visible execution) and 5 (hidden execution) are counted.
    """
    is_trade = messages["type"].isin([4, 5])
    signed_size = is_trade * messages["size"] * messages["direction"]

    times = messages["time"].values
    flow = np.zeros(len(messages), dtype=np.float64)

    # Two-pointer rolling sum
    left = 0
    running_sum = 0.0
    for right in range(len(messages)):
        running_sum += signed_size.iloc[right]
        while times[right] - times[left] > rolling_seconds:
            running_sum -= signed_size.iloc[left]
            left += 1
        flow[right] = running_sum

    return pd.Series(flow, index=messages.index, name=f"trade_flow_{rolling_seconds}s")


# ---------------------------------------------------------------------------
# Convenience: build full feature panel
# ---------------------------------------------------------------------------

def build_feature_panel(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    ofi_max_level: int = 5,
    trade_flow_windows: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> pd.DataFrame:
    """Construct the full feature panel for modeling.

    Combines book-derived features (OFI, depth imbalance, microprice deviation,
    spread) with trade-flow features over multiple time windows.

    Returns a DataFrame indexed the same as `orderbook`. NaN rows (from rolling
    windows at the start) are NOT dropped — callers should handle that.
    """
    features = pd.DataFrame(index=orderbook.index)

    # Multi-level OFI
    features = features.join(multi_level_ofi(orderbook, max_level=ofi_max_level))

    # Depth imbalances
    features["vol_imb_L1"] = volume_imbalance(orderbook, level=1)
    features["cum_depth_imb"] = cumulative_depth_imbalance(orderbook, max_level=ofi_max_level)

    # Price-derived
    features["micro_dev"] = micro_minus_mid(orderbook)
    features["spread_ticks"] = spread_in_ticks(orderbook)

    # Trade flow at multiple windows
    for w in trade_flow_windows:
        features[f"trade_flow_{w}s"] = trade_flow_imbalance(messages, rolling_seconds=w)

    return features


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

def make_targets(
    orderbook: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 30),
) -> pd.DataFrame:
    """Create classification targets for next-N-event price direction.

    For each horizon in `horizons`, computes:
        target_dir_hN = sign(mid_{t+N} - mid_t)
            with values {-1, 0, +1}

    No-move predictions (0) are kept in the target — downstream models can
    handle them as a third class or filter to non-zero.

    Returns a DataFrame indexed the same as `orderbook` with one column per horizon.
    """
    mid = compute_mid_price(orderbook)
    targets = pd.DataFrame(index=orderbook.index)
    for h in horizons:
        future_mid = mid.shift(-h)
        targets[f"target_dir_h{h}"] = np.sign(future_mid - mid)
    return targets
