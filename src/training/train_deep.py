"""Walk-forward training for LSTM and Transformer LOB direction classifiers.

Separate from train_baselines.py because the deep models consume (N, T, F)
sequences rather than flat (N, F) feature matrices, and they are too expensive
to refit at every fold under the same cadence used for linear/XGBoost.

Usage:
    python -m src.training.train_deep \\
        --ticker BTCUSDT \\
        --target-horizon 5 \\
        --seq-len 100 \\
        --models lstm transformer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.lobster import find_lobster_pair, load_paired
from src.data.features import build_feature_panel, make_targets
from src.models.deep_models import (
    LSTMBaseline, TransformerBaseline, build_lob_sequences,
)
from src.evaluation.metrics import classification_report


def prepare_sequence_dataset(
    data_dir: Path,
    ticker: str,
    target_horizon: int,
    seq_len: int,
    levels: int | None = None,
):
    files = find_lobster_pair(data_dir, ticker=ticker, levels=levels)
    print(f"Found {files.ticker} {files.date} levels={files.levels}")
    messages, orderbook = load_paired(files)

    features = build_feature_panel(messages, orderbook, ofi_max_level=files.levels)
    targets = make_targets(orderbook, horizons=(target_horizon,))

    combined = features.join(targets).dropna()
    feature_cols = list(features.columns)
    target_col = f"target_dir_h{target_horizon}"

    X_flat = combined[feature_cols].values.astype(np.float32)
    y_flat = combined[target_col].values.astype(np.int64)

    X_seq, y_seq = build_lob_sequences(X_flat, y_flat, seq_len=seq_len)
    print(f"Sequences: {X_seq.shape}; targets {len(y_seq):,}")
    return X_seq, y_seq, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--ticker", type=str, default="BTCUSDT")
    parser.add_argument("--levels", type=int, default=None)
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--initial-train-frac", type=float, default=0.5)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--refit-freq", type=int, default=0,
                        help="Samples between refits. 0 = single train on initial window, no refit.")
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer"])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = prepare_sequence_dataset(
        data_dir=data_dir,
        ticker=args.ticker,
        target_horizon=args.target_horizon,
        seq_len=args.seq_len,
        levels=args.levels,
    )

    n = len(X)
    n_train = int(n * args.initial_train_frac)
    n_test_per_fold = int(n * args.test_frac)
    min_gap = args.target_horizon

    print(f"Walk-forward setup: {n:,} samples, initial train {n_train:,}, test {n_test_per_fold:,} per fold")
    print(f"Class distribution overall: {pd.Series(y).value_counts().sort_index().to_dict()}")

    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")

        all_true = []
        all_pred = []
        fold_id = 0
        train_end = n_train

        while train_end + min_gap + n_test_per_fold <= n:
            # Fit on [0, train_end)
            if model_name == "lstm":
                model = LSTMBaseline(seq_len=args.seq_len, epochs=args.epochs)
            else:
                model = TransformerBaseline(seq_len=args.seq_len, epochs=args.epochs)
            model.fit(X[:train_end], y[:train_end])

            # Predict on [train_end + min_gap, train_end + min_gap + n_test_per_fold)
            test_start = train_end + min_gap
            test_end = test_start + n_test_per_fold
            y_pred = model.predict(X[test_start:test_end])
            y_true = y[test_start:test_end]

            all_true.append(y_true)
            all_pred.append(y_pred)

            acc = float((y_pred == y_true).mean())
            print(f"  fold {fold_id}: train_size={train_end}, test_size={n_test_per_fold}, accuracy={acc:.4f}")

            fold_id += 1
            train_end += n_test_per_fold

        # Pool predictions across folds for final report
        y_true_all = np.concatenate(all_true)
        y_pred_all = np.concatenate(all_pred)
        report = classification_report(y_true_all, y_pred_all)

        summary = {
            "model": model_name,
            "ticker": args.ticker,
            "target_horizon": args.target_horizon,
            "seq_len": args.seq_len,
            "n_folds": fold_id,
            "pooled_accuracy": report.accuracy,
            "pooled_directional_accuracy": report.directional_accuracy,
            "class_precision": report.class_precision,
            "class_recall": report.class_recall,
            "confusion": report.confusion.tolist(),
        }
        out_path = results_dir / "metrics" / f"{model_name}_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"  Pooled accuracy: {report.accuracy:.4f}")
        print(f"  Pooled directional accuracy: {report.directional_accuracy:.4f}")
        print(f"  Saved summary to {out_path}")


if __name__ == "__main__":
    main()
