"""Visualization utilities for limit order book analysis.

Plot types:

- Order book heatmap (price levels × time) showing depth evolution
- OFI signal over time with mid-price overlay
- Confusion matrix for direction prediction (3-class: -1, 0, +1)
- ROC / precision-recall by class
- Feature importance bar chart (for tree models)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 180,
})


def plot_mid_price(
    orderbook: pd.DataFrame,
    messages: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """Mid-price trajectory through the day."""
    from src.data.lobster import compute_mid_price, compute_microprice

    mid = compute_mid_price(orderbook)
    micro = compute_microprice(orderbook)
    time_hours = messages["time"] / 3600.0

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(time_hours, mid, color="steelblue", linewidth=0.6, label="Mid")
    ax.plot(time_hours, micro, color="crimson", linewidth=0.6, alpha=0.5, label="Microprice")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Price ($)")
    ax.set_title("Mid-price evolution")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_ofi_vs_mid(
    ofi: pd.Series,
    mid: pd.Series,
    window: int = 1000,
    save_path: Optional[Path] = None,
):
    """Rolling OFI signal alongside mid-price changes.

    Useful for visually confirming that aggregate OFI tracks mid moves.
    """
    rolling_ofi = ofi.rolling(window).sum()
    mid_change = mid.diff(window)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(rolling_ofi.values, color="steelblue", linewidth=0.5)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel(f"OFI ({window}-event rolling sum)")
    axes[0].set_title("Order flow imbalance and mid-price change")

    axes[1].plot(mid_change.values, color="crimson", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel(f"Mid change ({window} events)")
    axes[1].set_xlabel("Event index")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: list[int] = None,
    model_name: str = "model",
    save_path: Optional[Path] = None,
):
    """3-class confusion matrix heatmap for direction prediction {-1, 0, +1}."""
    if labels is None:
        labels = [-1, 0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues", aspect="auto")

    # Annotations
    n_classes = len(labels)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black")

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} confusion matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_decile_signal_check(
    feature: pd.Series,
    forward_return: pd.Series,
    feature_name: str = "OFI",
    save_path: Optional[Path] = None,
):
    """Decile-binned mean forward return — visual signal-check.

    Monotonic pattern = the feature has predictive value.
    """
    df = pd.DataFrame({"feat": feature, "ret": forward_return}).dropna()
    df["decile"] = pd.qcut(df["feat"], 10, labels=False, duplicates="drop")
    decile_means = df.groupby("decile")["ret"].mean() * 1e4  # in bps

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(decile_means.index, decile_means.values, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"{feature_name} decile (low → high)")
    ax.set_ylabel("Mean forward return (bps)")
    ax.set_title(f"Forward return by {feature_name} decile — monotonic = signal")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, ax
