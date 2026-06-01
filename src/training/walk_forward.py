"""Walk-forward cross-validation for time-series models.

Standard k-fold CV is invalid for time-series — it lets a model train on data
*after* the test point, leaking future information. Walk-forward CV respects
the time ordering: each fold trains on past data and tests on the next chunk.

Two modes supported:

1. **Expanding window:** training set grows with each fold
       Fold 1: train [t0, t1), test [t1, t1+W)
       Fold 2: train [t0, t2), test [t2, t2+W)
       ...

2. **Rolling window:** fixed training window size, sliding forward
       Fold 1: train [t0, t0+T), test [t0+T, t0+T+W)
       Fold 2: train [t0+W, t0+T+W), test [t0+T+W, t0+T+2W)
       ...

For LOB data: expanding window is usually preferred since more training data
is strictly better given non-stationarity is moderate within a day.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One walk-forward fold: indices for train and test."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train_idx)

    @property
    def test_size(self) -> int:
        return len(self.test_idx)


def expanding_window_splits(
    n_samples: int,
    initial_train_size: int,
    test_size: int,
    step: int | None = None,
    min_gap: int = 0,
) -> Iterator[Fold]:
    """Generate expanding-window walk-forward folds.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    initial_train_size : int
        Size of the first training window.
    test_size : int
        Number of samples in each test window.
    step : int, optional
        Step size between consecutive test windows. Defaults to `test_size`
        (non-overlapping test sets).
    min_gap : int
        Number of samples to skip between train and test (purges leakage from
        targets that look forward — set to your max prediction horizon).

    Yields
    ------
    Fold
    """
    if step is None:
        step = test_size

    train_end = initial_train_size
    fold_id = 0
    while train_end + min_gap + test_size <= n_samples:
        train_idx = np.arange(0, train_end)
        test_start = train_end + min_gap
        test_end = test_start + test_size
        test_idx = np.arange(test_start, test_end)

        yield Fold(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx)

        fold_id += 1
        train_end += step


def rolling_window_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    min_gap: int = 0,
) -> Iterator[Fold]:
    """Generate rolling-window walk-forward folds (fixed training size)."""
    if step is None:
        step = test_size

    train_start = 0
    fold_id = 0
    while train_start + train_size + min_gap + test_size <= n_samples:
        train_end = train_start + train_size
        train_idx = np.arange(train_start, train_end)
        test_start = train_end + min_gap
        test_end = test_start + test_size
        test_idx = np.arange(test_start, test_end)

        yield Fold(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx)

        fold_id += 1
        train_start += step


def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    initial_train_size: int,
    test_size: int,
    min_gap: int = 30,
    step: int | None = None,
    mode: str = "expanding",
) -> tuple[pd.DataFrame, list]:
    """Run walk-forward evaluation with a fresh model per fold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series (must be aligned with X).
    model_factory : callable
        Zero-arg callable that returns a fresh model with `.fit(X, y)` and
        `.predict(X)` methods (sklearn-compatible).
    initial_train_size : int
    test_size : int
    min_gap : int
        Buffer between train and test (set ≥ max prediction horizon).
    step : int, optional
        Step between folds.
    mode : str
        "expanding" or "rolling".

    Returns
    -------
    results : pd.DataFrame
        Per-fold metrics (accuracy, directional accuracy, fold_id, train_size, test_size).
    predictions : list[pd.DataFrame]
        Per-fold prediction DataFrames with (timestamp_idx, y_true, y_pred).
    """
    if mode == "expanding":
        folds = expanding_window_splits(
            n_samples=len(X),
            initial_train_size=initial_train_size,
            test_size=test_size,
            step=step,
            min_gap=min_gap,
        )
    elif mode == "rolling":
        folds = rolling_window_splits(
            n_samples=len(X),
            train_size=initial_train_size,
            test_size=test_size,
            step=step,
            min_gap=min_gap,
        )
    else:
        raise ValueError(f"Unknown mode {mode}")

    fold_results = []
    all_predictions = []

    for fold in folds:
        X_train = X.iloc[fold.train_idx]
        y_train = y.iloc[fold.train_idx]
        X_test = X.iloc[fold.test_idx]
        y_test = y.iloc[fold.test_idx]

        model = model_factory()
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)

        accuracy = float((y_pred == y_test.values).mean())
        # Directional accuracy on non-zero target moves only
        nonzero_mask = y_test.values != 0
        dir_acc = float((y_pred[nonzero_mask] == y_test.values[nonzero_mask]).mean()) if nonzero_mask.any() else np.nan

        fold_results.append({
            "fold_id": fold.fold_id,
            "train_size": fold.train_size,
            "test_size": fold.test_size,
            "accuracy": accuracy,
            "directional_accuracy": dir_acc,
        })
        all_predictions.append(pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred,
        }, index=X_test.index))

    return pd.DataFrame(fold_results), all_predictions
