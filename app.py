"""Machine Event Predictor — Streamlit showcase app (home page).

Run with:
    streamlit run app.py

The home page frames the business problem, summarizes the dataset and the
modelling approach, and points to the interactive pages in the sidebar.
"""

from __future__ import annotations

import streamlit as st

from pdm import config
from pdm import streamlit_utils as ui

ui.configure_page("Home")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🛠️ Machine Event Predictor")
st.markdown(
    "##### Predicting rare machine events from daily telemetry to optimize "
    "maintenance and prevent unplanned downtime."
)

trained = ui.model_is_trained()
if not trained:
    st.warning(
        "⚠️ The model has not been trained yet. Run **`python train.py`** in the "
        "project root to generate the artifacts, then refresh."
    )

meta = ui.load_metadata()
stats = meta.get("dataset_stats", {})
metrics = meta.get("metrics", {})

# --------------------------------------------------------------------------- #
# Dataset snapshot
# --------------------------------------------------------------------------- #
st.subheader("Dataset at a glance")
ui.metric_row([
    ("Records", f"{stats.get('n_records', 0):,}", "Daily machine telemetry rows"),
    ("Machines", f"{stats.get('n_machines', 0):,}", "Unique machines in the fleet"),
    ("Events", f"{stats.get('n_events', 0):,}", "Recorded machine events (the rare positive class)"),
    ("Event rate", f"{stats.get('event_rate', 0):.3%}",
     "Share of rows that are events — extreme class imbalance"),
])
if stats:
    st.caption(
        f"Daily records from **{stats.get('date_min')}** to **{stats.get('date_max')}** · "
        f"imbalance ratio ≈ **1 : {round(1 / max(stats.get('event_rate', 1e-9), 1e-9)):,}**"
    )

st.divider()

# --------------------------------------------------------------------------- #
# Business context + approach
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

with left:
    st.subheader("🏭 The business problem")
    st.markdown(
        """
        The customer runs a **fleet of machines** that transmit daily aggregated
        telemetry. A *machine event* can lead to maintenance issues, downtime,
        and suboptimal provisioning if it isn't anticipated.

        The goal is to **predict event occurrences in advance** so maintenance
        can be scheduled proactively.

        **Why it's hard**
        - Events are *extremely rare* (well under 1% of records).
        - **False negatives** are costly — they mean unplanned downtime.
        - **False positives** waste maintenance effort, so they must be controlled.
        - Plain accuracy is misleading: always predicting "no event" scores ~99.9%.
        """
    )

with right:
    st.subheader("🧠 The modelling approach")
    st.markdown(
        f"""
        1. **Temporal features** — extract `day_of_week`, `day_of_month`, and
           `month` from the date to capture time patterns.
        2. **Anomaly score** — an **Isolation Forest** flags how unusual each
           daily record is; its score becomes an extra feature (`iso_score`).
        3. **Cost-sensitive XGBoost** — a gradient-boosted tree classifier with
           `scale_pos_weight` to handle the imbalance.
        4. **Threshold tuning** — the decision threshold is lowered to
           **{config.DEFAULT_THRESHOLD}** (not 0.5) to balance the cost of false
           negatives vs. false positives.

        Models are evaluated with **Average Precision**, which summarizes the
        precision–recall trade-off — the right lens for rare-event detection.
        """
    )

st.divider()

# --------------------------------------------------------------------------- #
# Headline performance
# --------------------------------------------------------------------------- #
st.subheader("Headline model performance (held-out test set)")
if metrics:
    ap = metrics.get("average_precision", 0)
    baseline = metrics.get("random_baseline_ap", 0)
    lift = ap / baseline if baseline else 0
    ui.metric_row([
        ("Average Precision", f"{ap:.3f}",
         f"≈ {lift:.0f}× better than the random baseline of {baseline:.5f}"),
        ("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", "Ranking quality across all thresholds"),
        ("Recall @ threshold", f"{metrics.get('recall', 0):.1%}", "Share of true events caught"),
        ("Precision @ threshold", f"{metrics.get('precision', 0):.1%}",
         "Share of alerts that are real events"),
    ])
else:
    st.info("Train the model to see performance metrics here.")

st.divider()

# --------------------------------------------------------------------------- #
# Navigation guide
# --------------------------------------------------------------------------- #
st.subheader("Explore the app")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        - **📊 Data Explorer** — distributions, event timeline, per-machine drill-down.
        - **🔮 Event Simulator** — feed in telemetry and watch the model react in real time.
        """
    )
with c2:
    st.markdown(
        """
        - **🚨 Fleet Monitoring** — rank the fleet by risk and triage today's alerts.
        - **📈 Model Performance** — interactive threshold tuning, PR curve, feature importance.
        """
    )

st.caption("Use the sidebar to navigate between pages.")

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Showcase app for a predictive-maintenance model that detects rare "
        "machine events from daily telemetry."
    )
    if meta.get("trained_at"):
        st.caption(f"Model trained: {meta['trained_at']}")
