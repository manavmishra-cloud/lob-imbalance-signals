"""Evaluation metrics for short-horizon price direction prediction.

Beyond raw accuracy, useful metrics for LOB prediction include:

- **Directional accuracy** on non-zero target moves (filters out no-move events)
- **Confusion matrix** with class-wise precision/recall
- **Strategy-PnL Sharpe** for an idealized paper-trading rule
- **Calibration** (Brier score, reliability diagram)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class ClassificationReport:
    accuracy: float
    directional_accuracy: float  # accuracy on non-zero target moves only
    class_precision: dict
    class_recall: dict
    confusion: np.ndarray


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:
    """Comprehensive classification breakdown for {-1, 0, +1} targets."""
    accuracy = float((y_pred == y_true).mean())

    nonzero = y_true != 0
    dir_acc = float((y_pred[nonzero] == y_true[nonzero]).mean()) if nonzero.any() else float("nan")

    labels = [-1, 0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision = {}
    recall = {}
    for i, lbl in enumerate(labels):
        col_sum = cm[:, i].sum()
        row_sum = cm[i, :].sum()
        precision[lbl] = float(cm[i, i] / col_sum) if col_sum > 0 else float("nan")
        recall[lbl] = float(cm[i, i] / row_sum) if row_sum > 0 else float("nan")

    return ClassificationReport(
        accuracy=accuracy,
        directional_accuracy=dir_acc,
        class_precision=precision,
        class_recall=recall,
        confusion=cm,
    )


def strategy_sharpe(
    y_pred: np.ndarray,
    mid_returns: np.ndarray,
    threshold_prob: Optional[float] = None,
    pred_proba: Optional[np.ndarray] = None,
    annualization_factor: float = 252.0 * 6.5 * 3600,
) -> float:
    """Sharpe ratio of an idealized strategy that takes a position based on `y_pred`.

    Rule: if model predicts +1, go long for that bar; if -1, go short; if 0, flat.
    PnL of each trade = mid_return * position.

    Parameters
    ----------
    y_pred : array of {-1, 0, +1}
    mid_returns : array of forward mid-price returns at the same horizon as the target
    threshold_prob : float, optional
        If provided alongside `pred_proba`, only trade when max predicted probability
        exceeds this threshold (filters low-confidence predictions).
    pred_proba : array of shape (n, 3), optional
        Per-class probabilities ordered as [-1, 0, +1].
    annualization_factor : float
        Annualization scaling. Default assumes events arrive ~1 per second.
        For seconds: 252 trading days * 6.5 hours * 3600 seconds = ~5.9M.
        Adjust based on your event frequency.

    Returns
    -------
    Annualized Sharpe ratio.
    """
    positions = y_pred.astype(float)
    if threshold_prob is not None and pred_proba is not None:
        max_proba = pred_proba.max(axis=1)
        positions = np.where(max_proba >= threshold_prob, positions, 0.0)

    pnl = positions * mid_returns
    if pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(annualization_factor))


def aggregate_walk_forward_metrics(fold_results: pd.DataFrame) -> dict:
    """Summarize walk-forward CV results across folds."""
    return {
        "n_folds": len(fold_results),
        "accuracy_mean": float(fold_results["accuracy"].mean()),
        "accuracy_std": float(fold_results["accuracy"].std()),
        "directional_accuracy_mean": float(fold_results["directional_accuracy"].mean()),
        "directional_accuracy_std": float(fold_results["directional_accuracy"].std()),
        "total_train_size_final": int(fold_results["train_size"].max()),
        "test_size_per_fold": int(fold_results["test_size"].iloc[0]) if len(fold_results) > 0 else 0,
    }
