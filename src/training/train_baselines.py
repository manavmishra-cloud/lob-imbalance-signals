"""End-to-end training script for the LOB baseline models.

Usage:
    python -m src.training.train_baselines \\
        --data-dir data/raw \\
        --ticker AAPL \\
        --target-horizon 5 \\
        --initial-train-frac 0.5 \\
        --test-frac 0.1

Outputs:
    - results/metrics/{model}_walkforward.csv  (per-fold metrics)
    - results/metrics/{model}_summary.json     (aggregated summary)
    - results/figures/{model}_confusion.png    (confusion matrix plot)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.lobster import find_lobster_pair, load_paired
from src.data.features import build_feature_panel, make_targets
from src.models.baselines import LinearOFIBaseline, PersistenceBaseline, XGBoostBaseline
from src.training.walk_forward import walk_forward_evaluate
from src.evaluation.metrics import classification_report, aggregate_walk_forward_metrics


def prepare_dataset(
    data_dir: Path,
    ticker: str,
    target_horizon: int,
    levels: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Locate LOBSTER files, build features, and align with target.

    Returns
    -------
    X : feature panel (NaN-purged)
    y : target series for the requested horizon
    raw : full feature+target DataFrame for inspection
    """
    files = find_lobster_pair(data_dir, ticker=ticker, levels=levels)
    print(f"Found LOBSTER pair: {files.ticker} {files.date} levels={files.levels}")

    messages, orderbook = load_paired(files)
    print(f"Loaded {len(messages):,} events")

    features = build_feature_panel(messages, orderbook, ofi_max_level=files.levels)
    targets = make_targets(orderbook, horizons=(target_horizon,))

    combined = features.join(targets).dropna()
    X = combined[features.columns]
    y = combined[f"target_dir_h{target_horizon}"].astype(int)

    print(f"After NaN purge: {len(X):,} samples, {X.shape[1]} features")
    print(f"Target class distribution:\n{y.value_counts().sort_index()}")
    return X, y, combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--levels", type=int, default=None, help="Book depth (1, 5, 10, 30, 50)")
    parser.add_argument("--target-horizon", type=int, default=5, help="Prediction horizon in events")
    parser.add_argument("--initial-train-frac", type=float, default=0.5)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--models", nargs="+", default=["persistence", "linear", "xgboost"])
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ---- Build dataset ----
    X, y, _ = prepare_dataset(
        data_dir=data_dir,
        ticker=args.ticker,
        target_horizon=args.target_horizon,
        levels=args.levels,
    )

    # ---- Walk-forward setup ----
    n = len(X)
    initial_train_size = int(n * args.initial_train_frac)
    test_size = int(n * args.test_frac)
    min_gap = args.target_horizon  # Purge label leakage

    model_factories = {
        "persistence": lambda: PersistenceBaseline(),
        "linear": lambda: LinearOFIBaseline(),
        "xgboost": lambda: XGBoostBaseline(),
    }

    summary = {}

    for model_name in args.models:
        if model_name not in model_factories:
            print(f"Unknown model '{model_name}', skipping.")
            continue

        print(f"\n=== Training {model_name} ===")
        fold_results, predictions = walk_forward_evaluate(
            X=X,
            y=y,
            model_factory=model_factories[model_name],
            initial_train_size=initial_train_size,
            test_size=test_size,
            min_gap=min_gap,
            mode="expanding",
        )

        # Save per-fold results
        fold_path = results_dir / "metrics" / f"{model_name}_walkforward.csv"
        fold_results.to_csv(fold_path, index=False)
        print(f"  Saved fold results to {fold_path}")

        # Aggregate
        agg = aggregate_walk_forward_metrics(fold_results)
        agg["model"] = model_name
        agg["target_horizon"] = args.target_horizon
        agg["ticker"] = args.ticker

        # Combined confusion across all test predictions
        all_true = np.concatenate([p["y_true"].values for p in predictions])
        all_pred = np.concatenate([p["y_pred"].values for p in predictions])
        report = classification_report(all_true, all_pred)
        agg["pooled_accuracy"] = report.accuracy
        agg["pooled_directional_accuracy"] = report.directional_accuracy
        agg["pooled_confusion"] = report.confusion.tolist()
        agg["pooled_class_precision"] = report.class_precision
        agg["pooled_class_recall"] = report.class_recall

        summary_path = results_dir / "metrics" / f"{model_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(agg, f, indent=2, default=float)
        print(f"  Saved summary to {summary_path}")
        print(f"  Mean accuracy: {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
        print(f"  Mean directional accuracy: {agg['directional_accuracy_mean']:.4f}")

        summary[model_name] = agg

    # ---- Cross-model summary ----
    print("\n=== Summary across models ===")
    summary_df = pd.DataFrame([
        {
            "model": m,
            "accuracy_mean": v["accuracy_mean"],
            "accuracy_std": v["accuracy_std"],
            "directional_accuracy_mean": v["directional_accuracy_mean"],
            "pooled_accuracy": v["pooled_accuracy"],
        }
        for m, v in summary.items()
    ])
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / "metrics" / "all_models_summary.csv", index=False)


if __name__ == "__main__":
    main()
