"""Central configuration: paths, feature definitions, and model hyperparameters.

Everything that the notebook hard-codes is collected here so the training
script and the Streamlit app stay in lock-step. Changing a value in one
place propagates to both.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "dataset" / "machine_event.csv"
MODELS_DIR = ROOT_DIR / "models"

XGB_MODEL_PATH = MODELS_DIR / "xgb_model.json"
ISO_FOREST_PATH = MODELS_DIR / "iso_forest.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# --------------------------------------------------------------------------- #
# Data schema
# --------------------------------------------------------------------------- #
CSV_SEP = ";"
DATE_FORMAT = "%m/%d/%Y"  # US month-first format used in the raw CSV
DATE_COL = "date"
MACHINE_COL = "machine"
TARGET_COL = "event"

# Raw telemetry features as they appear in the CSV.
RAW_FEATURES = [f"feature{i}" for i in range(1, 10)]

# Temporal features engineered from the date field.
DATE_FEATURES = ["day_of_week", "day_of_month", "month"]

# Anomaly score produced by the Isolation Forest.
ANOMALY_FEATURE = "iso_score"

# Features fed to the Isolation Forest (everything except the anomaly score).
BASE_FEATURES = RAW_FEATURES + DATE_FEATURES

# Final feature order consumed by the XGBoost classifier. Order matters:
# XGBoost matches columns by position when given a numpy array, so the app
# must build inference rows in exactly this order.
MODEL_FEATURES = BASE_FEATURES + [ANOMALY_FEATURE]

# --------------------------------------------------------------------------- #
# Model hyperparameters (taken from the notebook's Grid Search CV results)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2

ISO_FOREST_PARAMS = {
    "contamination": 0.001,
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "aucpr",
}

# Decision threshold selected in the notebook by maximizing F1 on the test set.
# The default 0.5 is inappropriate for such an imbalanced problem.
DEFAULT_THRESHOLD = 0.022

# --------------------------------------------------------------------------- #
# Risk banding for the UI (applied to predicted event probability)
# --------------------------------------------------------------------------- #
RISK_BANDS = [
    # (label, lower_bound_inclusive, color)
    ("Low", 0.0, "#2ecc71"),
    ("Elevated", DEFAULT_THRESHOLD, "#f1c40f"),
    ("High", 0.10, "#e67e22"),
    ("Critical", 0.30, "#e74c3c"),
]


def risk_band(prob: float) -> tuple[str, str]:
    """Return the (label, color) risk band for a given event probability."""
    label, color = RISK_BANDS[0][0], RISK_BANDS[0][2]
    for band_label, lower, band_color in RISK_BANDS:
        if prob >= lower:
            label, color = band_label, band_color
    return label, color
