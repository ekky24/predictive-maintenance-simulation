"""Train the machine-event model and persist artifacts to ``models/``.

Reproduces the notebook end-to-end:
  * stratified 80/20 train-test split,
  * Isolation Forest anomaly score,
  * cost-sensitive XGBoost,
  * F1-optimal decision threshold,
  * evaluation metrics saved for the Streamlit "Model Performance" page.

Usage:
    python train.py            # train with the notebook's tuned threshold
    python train.py --retune   # re-search the F1-optimal threshold on test
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pdm import config, data, model as model_mod


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return the threshold that maximizes F1 over a fine grid (notebook logic)."""
    grid = np.linspace(0, 1, 500)
    best_t, best_f1 = config.DEFAULT_THRESHOLD, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main(retune: bool) -> None:
    print("Loading dataset ...")
    df = data.load_dataset()
    X = df[config.BASE_FEATURES]
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"  train={X_train.shape[0]:,}  test={X_test.shape[0]:,}  "
          f"event-rate={y.mean():.4%}")

    print("Training Isolation Forest + XGBoost ...")
    me_model = model_mod.fit(X_train, y_train)

    # --- Evaluate on the held-out test set ------------------------------- #
    y_proba = me_model.predict_proba(X_test)

    threshold = tune_threshold(y_test.to_numpy(), y_proba) if retune else config.DEFAULT_THRESHOLD
    me_model.threshold = threshold
    y_pred = (y_proba >= threshold).astype(int)

    ap = float(average_precision_score(y_test, y_proba))
    roc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()
    prec, rec, thr = precision_recall_curve(y_test, y_proba)

    metrics = {
        "average_precision": ap,
        "roc_auc": roc,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "random_baseline_ap": float(y.mean()),
    }

    # Down-sample PR curve to keep metadata compact.
    pr_idx = np.linspace(0, len(prec) - 1, num=min(300, len(prec))).astype(int)
    pr_curve = {
        "precision": prec[pr_idx].tolist(),
        "recall": rec[pr_idx].tolist(),
    }

    dataset_stats = {
        "n_records": int(len(df)),
        "n_machines": int(df[config.MACHINE_COL].nunique()),
        "n_events": int(y.sum()),
        "event_rate": float(y.mean()),
        "date_min": str(df[config.DATE_COL].min().date()),
        "date_max": str(df[config.DATE_COL].max().date()),
    }

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "scale_pos_weight": model_mod.imbalance_ratio(y_train),
        "xgb_params": config.XGB_PARAMS,
        "iso_forest_params": config.ISO_FOREST_PARAMS,
        "metrics": metrics,
        "pr_curve": pr_curve,
        "feature_importance": me_model.feature_importance().to_dict("records"),
        "dataset_stats": dataset_stats,
    }

    me_model.save(metadata=metadata)

    # Persist test-set predictions for the interactive performance page.
    test_out = pd.DataFrame({
        "machine": df.loc[X_test.index, config.MACHINE_COL].to_numpy(),
        "date": df.loc[X_test.index, config.DATE_COL].to_numpy(),
        "y_true": y_test.to_numpy(),
        "y_proba": y_proba,
    })
    test_out.to_parquet(config.MODELS_DIR / "test_predictions.parquet", index=False)

    print("\n=== Results ===")
    print(f"  Average Precision : {ap:.4f}  (random baseline {y.mean():.5f})")
    print(f"  ROC-AUC           : {roc:.4f}")
    print(f"  Threshold         : {threshold:.4f}")
    print(f"  Precision/Recall  : {metrics['precision']:.3f} / {metrics['recall']:.3f}")
    print(f"  Confusion matrix  : {cm}")
    print(f"\nArtifacts written to {config.MODELS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune", action="store_true",
                        help="re-search the F1-optimal threshold on the test set")
    args = parser.parse_args()
    main(retune=args.retune)
