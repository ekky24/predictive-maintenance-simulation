"""Fleet Monitoring — score the whole fleet and triage the riskiest machines.

Picks a snapshot day (or the full history), ranks machines by predicted event
probability, and lets operators tune the alert threshold to balance how many
machines get flagged against how many true events are caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pdm import config
from pdm import streamlit_utils as ui

ui.configure_page("Fleet Monitoring")
st.title("🚨 Fleet Monitoring")
st.caption("Rank the fleet by predicted risk and triage the day's alerts.")

model = ui.require_model()
df = ui.load_data()


@st.cache_data(show_spinner="Scoring the full fleet …")
def score_fleet() -> pd.DataFrame:
    """Score every record once: event probability + anomaly score."""
    base = df[config.BASE_FEATURES]
    scored = df[[config.DATE_COL, config.MACHINE_COL, config.TARGET_COL]].copy()
    scored["probability"] = model.predict_proba(base)
    scored["anomaly_score"] = model.anomaly_score(base)
    return scored


scored = score_fleet()

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
ctrl1, ctrl2 = st.columns([1, 1])
with ctrl1:
    scope = st.radio("Scope", ["Single day snapshot", "Full history"], horizontal=True)
with ctrl2:
    threshold = st.slider(
        "Alert threshold", 0.0, 0.5, float(model.threshold), step=0.001, format="%.3f",
        help=f"Flag a machine when probability ≥ threshold. Tuned default = {model.threshold}.",
    )

if scope == "Single day snapshot":
    available_days = scored[config.DATE_COL].dt.date
    snapshot = st.select_slider(
        "Snapshot day",
        options=sorted(available_days.unique()),
        value=sorted(available_days.unique())[-1],
    )
    view = scored[scored[config.DATE_COL].dt.date == snapshot].copy()
    scope_label = f"on {snapshot}"
else:
    view = scored.copy()
    scope_label = "across full history"

view["flagged"] = view["probability"] >= threshold

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
n_records = len(view)
n_flagged = int(view["flagged"].sum())
n_events = int(view[config.TARGET_COL].sum())
caught = int(view[view["flagged"]][config.TARGET_COL].sum())
recall = caught / n_events if n_events else 0.0
precision = caught / n_flagged if n_flagged else 0.0

ui.metric_row([
    ("Records scored", f"{n_records:,}", f"Machines evaluated {scope_label}"),
    ("Alerts raised", f"{n_flagged:,}",
     f"{n_flagged / n_records:.2%} of records flagged" if n_records else None),
    ("Events caught", f"{caught} / {n_events}",
     f"Recall {recall:.0%}" if n_events else "No actual events in this slice"),
    ("Alert precision", f"{precision:.1%}" if n_flagged else "—",
     "Share of alerts that are real events"),
])

st.divider()

# --------------------------------------------------------------------------- #
# Risk leaderboard
# --------------------------------------------------------------------------- #
left, right = st.columns([1.4, 1])

with left:
    st.subheader("🔝 Highest-risk records")
    top_n = st.slider("Show top N", 5, 50, 15)
    ranked = view.sort_values("probability", ascending=False).head(top_n).copy()
    ranked["risk"] = ranked["probability"].apply(lambda p: config.risk_band(p)[0])
    ranked["actual event"] = ranked[config.TARGET_COL].map({0: "", 1: "✅ yes"})
    display = ranked[[
        config.MACHINE_COL, config.DATE_COL, "probability",
        "anomaly_score", "risk", "actual event",
    ]].rename(columns={config.MACHINE_COL: "machine", config.DATE_COL: "date"})
    st.dataframe(
        display,
        hide_index=True, use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn("date"),
            "probability": st.column_config.ProgressColumn(
                "probability", min_value=0.0,
                max_value=float(display["probability"].max() or 1.0), format="%.4f",
            ),
            "anomaly_score": st.column_config.NumberColumn("anomaly", format="%.3f"),
        },
    )
    st.download_button(
        "⬇️ Download full ranked list (CSV)",
        view.sort_values("probability", ascending=False).to_csv(index=False),
        file_name="fleet_risk_ranking.csv", mime="text/csv",
    )

with right:
    st.subheader("Probability distribution")
    fig = px.histogram(
        view, x="probability", nbins=60, log_y=True,
        title="Predicted probabilities (log count)",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#e74c3c",
                  annotation_text="threshold", annotation_position="top")
    fig.update_traces(marker_color="#3498db")
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Alert breakdown")
    breakdown = pd.DataFrame({
        "outcome": ["True alert (event)", "False alarm", "Missed event"],
        "count": [caught, n_flagged - caught, n_events - caught],
    })
    fig2 = px.bar(
        breakdown, x="count", y="outcome", orientation="h", text="count",
        color="outcome",
        color_discrete_map={
            "True alert (event)": "#2ecc71",
            "False alarm": "#f1c40f",
            "Missed event": "#e74c3c",
        },
    )
    fig2.update_layout(showlegend=False, height=240, margin=dict(t=10, b=10), yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

st.info(
    "💡 Lowering the threshold catches more true events (higher recall) at the cost "
    "of more false alarms (lower precision). Use the **Model Performance** page to "
    "see the full trade-off curve."
)
