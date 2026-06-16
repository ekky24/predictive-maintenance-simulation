"""Streamlit helpers shared across pages: cached loaders and small UI bits.

Kept separate from the Streamlit ``pages/`` directory so it is *not* picked up
as a navigable page. Heavy objects (model, dataset, test predictions) are
cached with ``st.cache_*`` so they load once per session.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import config, data
from .model import MachineEventModel

PAGE_ICON = "🛠️"


def configure_page(title: str, layout: str = "wide") -> None:
    """Apply consistent ``st.set_page_config`` settings for every page."""
    st.set_page_config(
        page_title=f"{title} · Machine Event Predictor",
        page_icon=PAGE_ICON,
        layout=layout,
        initial_sidebar_state="expanded",
    )


@st.cache_resource(show_spinner="Loading trained model …")
def load_model() -> MachineEventModel:
    """Load the trained model bundle (cached for the whole session)."""
    return MachineEventModel.load()


@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    """Load the dataset with engineered features (cached)."""
    return data.load_dataset()


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Read the training metadata sidecar (metrics, stats, importances)."""
    return MachineEventModel.metadata()


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    """Load held-out test-set predictions saved during training."""
    path = config.MODELS_DIR / "test_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["machine", "date", "y_true", "y_proba"])


def model_is_trained() -> bool:
    """True if the trained artifacts exist on disk."""
    return config.XGB_MODEL_PATH.exists()


def require_model() -> MachineEventModel | None:
    """Load the model or render a friendly 'train me first' message."""
    if not model_is_trained():
        st.error(
            "No trained model found. Run **`python train.py`** in the project "
            "root to build the model artifacts, then refresh this page."
        )
        st.stop()
    return load_model()


def risk_badge(prob: float) -> str:
    """Return an HTML pill summarizing the risk band for a probability."""
    label, color = config.risk_band(prob)
    return (
        f"<span style='background:{color};color:#111;padding:4px 12px;"
        f"border-radius:12px;font-weight:600;font-size:0.9rem;'>"
        f"{label} risk</span>"
    )


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of ``st.metric`` widgets from (label, value, help) tuples."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)
