"""Model Performance — interactive threshold tuning, PR curve, feature importance.

All interactive metrics are recomputed from the held-out test-set predictions
saved during training, so moving the threshold slider instantly shows the
precision/recall/confusion-matrix trade-off without re-running the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from pdm import config
from pdm import streamlit_utils as ui

ui.configure_page("Model Performance")
st.title("📈 Model Performance")
st.caption("Evaluate the model on the held-out test set and explore the precision–recall trade-off.")

model = ui.require_model()
meta = ui.load_metadata()
preds = ui.load_test_predictions()

if preds.empty:
    st.error("No saved test predictions found. Re-run `python train.py`.")
    st.stop()

y_true = preds["y_true"].to_numpy()
y_proba = preds["y_proba"].to_numpy()

ap = average_precision_score(y_true, y_proba)
roc = roc_auc_score(y_true, y_proba)
baseline = float(y_true.mean())

# --------------------------------------------------------------------------- #
# Threshold-independent headline metrics
# --------------------------------------------------------------------------- #
st.subheader("Threshold-independent metrics")
ui.metric_row([
    ("Average Precision", f"{ap:.3f}", "Area under the precision–recall curve"),
    ("Random baseline AP", f"{baseline:.5f}", "AP of guessing at the event rate"),
    ("Lift over baseline", f"{ap / baseline:.0f}×", "How much better than random"),
    ("ROC-AUC", f"{roc:.3f}", "Ranking quality across all thresholds"),
])
st.caption(
    f"Test set: {len(y_true):,} records · {int(y_true.sum())} actual events "
    f"({baseline:.3%} event rate)."
)

st.divider()

# --------------------------------------------------------------------------- #
# Interactive threshold
# --------------------------------------------------------------------------- #
st.subheader("🎚️ Interactive threshold tuning")
st.markdown(
    "The model outputs a *probability*. The **threshold** converts it to a decision. "
    "Because events are rare and false negatives are costly, the threshold is set well "
    "below the naive 0.5."
)

threshold = st.slider(
    "Decision threshold", 0.0, 0.5, float(model.threshold), step=0.001, format="%.3f",
)
y_pred = (y_proba >= threshold).astype(int)

prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

ui.metric_row([
    ("Precision", f"{prec:.1%}", "Of flagged records, how many are real events"),
    ("Recall", f"{rec:.1%}", "Of real events, how many were caught"),
    ("F1 score", f"{f1:.3f}", "Harmonic mean of precision and recall"),
    ("Alerts raised", f"{int(tp + fp):,}", "Total records flagged at this threshold"),
])

cm_col, note_col = st.columns([1, 1])
with cm_col:
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Event", "Actual: Event"],
        columns=["Predicted: No Event", "Predicted: Event"],
    )
    fig = px.imshow(
        cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        title=f"Confusion matrix @ threshold {threshold:.3f}",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=360, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with note_col:
    st.markdown("**What this threshold means operationally**")
    st.markdown(
        f"""
        - ✅ **{tp}** true events caught (true positives)
        - 🚨 **{fp}** false alarms (false positives) — wasted maintenance checks
        - ❌ **{fn}** missed events (false negatives) — risk of unplanned downtime
        - ⚪ **{tn:,}** correctly cleared normal records

        At threshold **{threshold:.3f}**, the team investigates **{int(tp + fp)}**
        records to catch **{tp}** of **{int(tp + fn)}** events.
        """
    )
    if threshold != model.threshold:
        st.caption(f"Tuned default threshold is {model.threshold}.")

st.divider()

# --------------------------------------------------------------------------- #
# PR curve & ROC curve
# --------------------------------------------------------------------------- #
st.subheader("Trade-off curves")
pr_p, pr_r, pr_t = precision_recall_curve(y_true, y_proba)
fpr, tpr, _ = roc_curve(y_true, y_proba)

# Operating point at the current threshold.
op_recall = rec
op_precision = prec

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr_r, y=pr_p, mode="lines", name="PR curve",
                             line=dict(color="#3498db")))
    fig.add_hline(y=baseline, line_dash="dot", line_color="grey",
                  annotation_text="random baseline")
    fig.add_trace(go.Scatter(
        x=[op_recall], y=[op_precision], mode="markers",
        marker=dict(color="#e74c3c", size=12, symbol="x"),
        name="current threshold",
    ))
    fig.update_layout(
        title=f"Precision–Recall (AP = {ap:.3f})",
        xaxis_title="Recall", yaxis_title="Precision",
        height=380, margin=dict(t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve",
                             line=dict(color="#9b59b6")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dot", color="grey"), name="random"))
    fig.update_layout(
        title=f"ROC curve (AUC = {roc:.3f})",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=380, margin=dict(t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
st.subheader("Feature importance")
st.caption(
    "Which signals drive the model. Note the engineered features — the Isolation "
    "Forest `iso_score` and the temporal features — earning their place alongside raw telemetry."
)
imp = pd.DataFrame(meta.get("feature_importance", []))
if not imp.empty:
    imp = imp.sort_values("importance", ascending=True)
    colors = ["#e67e22" if f in (config.DATE_FEATURES + [config.ANOMALY_FEATURE])
              else "#3498db" for f in imp["feature"]]
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 title="XGBoost feature importance")
    fig.update_traces(marker_color=colors)
    fig.update_layout(height=460, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟠 Engineered features (anomaly score + temporal) · 🔵 raw telemetry")
else:
    st.info("Feature importance unavailable — re-run training.")
