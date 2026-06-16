"""Model definition, training, persistence, and inference.

The pipeline reproduces the notebook:

    raw features + temporal features
        -> Isolation Forest anomaly score (iso_score)
        -> XGBoost classifier (cost-sensitive via scale_pos_weight)
        -> probability -> threshold -> event / no-event

``MachineEventModel`` bundles the two fitted estimators plus the decision
threshold so the app can run end-to-end inference from a single object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from . import config


@dataclass
class MachineEventModel:
    """Trained Isolation Forest + XGBoost bundle with a decision threshold."""

    iso_forest: IsolationForest
    classifier: XGBClassifier
    threshold: float = config.DEFAULT_THRESHOLD
    feature_names: list[str] = field(default_factory=lambda: list(config.MODEL_FEATURES))

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def anomaly_score(self, base_features: pd.DataFrame) -> np.ndarray:
        """Compute the Isolation Forest anomaly score.

        Higher means *more* anomalous. We negate ``decision_function`` so the
        score increases with abnormality (matching the notebook's ``iso_score``).
        """
        base = base_features[config.BASE_FEATURES]
        return -self.iso_forest.decision_function(base)

    def build_matrix(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Assemble the full 13-column feature matrix (base + iso_score)."""
        matrix = base_features[config.BASE_FEATURES].copy()
        matrix[config.ANOMALY_FEATURE] = self.anomaly_score(base_features)
        return matrix[self.feature_names]

    def predict_proba(self, base_features: pd.DataFrame) -> np.ndarray:
        """Return the predicted probability of a machine event."""
        matrix = self.build_matrix(base_features)
        return self.classifier.predict_proba(matrix)[:, 1]

    def predict(self, base_features: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Return binary event predictions using the (optional) threshold."""
        thr = self.threshold if threshold is None else threshold
        return (self.predict_proba(base_features) >= thr).astype(int)

    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted feature-importance table (XGBoost gain-weighted)."""
        importances = self.classifier.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, metadata: dict | None = None) -> None:
        """Persist the estimators and a metadata sidecar to ``models/``."""
        import joblib

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(config.XGB_MODEL_PATH)
        joblib.dump(self.iso_forest, config.ISO_FOREST_PATH)

        meta = {
            "threshold": self.threshold,
            "feature_names": self.feature_names,
        }
        if metadata:
            meta.update(metadata)
        config.METADATA_PATH.write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls) -> "MachineEventModel":
        """Load a previously trained model from ``models/``."""
        import joblib

        if not config.XGB_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {config.XGB_MODEL_PATH}. "
                "Run `python train.py` first."
            )

        classifier = XGBClassifier()
        classifier.load_model(config.XGB_MODEL_PATH)
        iso_forest = joblib.load(config.ISO_FOREST_PATH)

        meta = json.loads(config.METADATA_PATH.read_text())
        return cls(
            iso_forest=iso_forest,
            classifier=classifier,
            threshold=float(meta.get("threshold", config.DEFAULT_THRESHOLD)),
            feature_names=meta.get("feature_names", list(config.MODEL_FEATURES)),
        )

    @staticmethod
    def metadata() -> dict:
        """Read the metadata sidecar (metrics, training stats, etc.)."""
        if config.METADATA_PATH.exists():
            return json.loads(config.METADATA_PATH.read_text())
        return {}


def imbalance_ratio(y) -> float:
    """Negative-to-positive class ratio, used for cost-sensitive learning."""
    y = np.asarray(y)
    pos = max((y == 1).sum(), 1)
    return float((y == 0).sum() / pos)


def fit(X_train: pd.DataFrame, y_train: pd.Series, threshold: float = config.DEFAULT_THRESHOLD) -> MachineEventModel:
    """Train the full pipeline on the given training split.

    ``X_train`` must contain ``config.BASE_FEATURES``. The Isolation Forest is
    fit first, its score appended, then XGBoost is trained on all 13 features
    with ``scale_pos_weight`` set to the class imbalance ratio.
    """
    iso_forest = IsolationForest(**config.ISO_FOREST_PARAMS)
    iso_forest.fit(X_train[config.BASE_FEATURES])

    matrix = X_train[config.BASE_FEATURES].copy()
    matrix[config.ANOMALY_FEATURE] = -iso_forest.decision_function(X_train[config.BASE_FEATURES])
    matrix = matrix[config.MODEL_FEATURES]

    classifier = XGBClassifier(
        scale_pos_weight=imbalance_ratio(y_train),
        **config.XGB_PARAMS,
    )
    classifier.fit(matrix, y_train)

    return MachineEventModel(iso_forest=iso_forest, classifier=classifier, threshold=threshold)
