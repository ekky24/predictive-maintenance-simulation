"""Event Simulator — feed telemetry into the model and watch it react.

Users can start from a real record in the dataset (any machine/day) or dial in
values by hand, then see the predicted event probability, the anomaly score,
and how the decision changes with the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm import config
from pdm import streamlit_utils as ui

ui.configure_page("Event Simulator")
st.title("🔮 Event Simulator")
st.caption("Adjust a machine's daily telemetry and see the model's live risk assessment.")

model = ui.require_model()
df = ui.load_data()

# Percentiles used to set reasonable slider ranges and to contextualize inputs.
feat_stats = df[config.RAW_FEATURES].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T

# --------------------------------------------------------------------------- #
# Seed values — optionally load a real record into the controls
# --------------------------------------------------------------------------- #
st.sidebar.header("Starting point")
seed_mode = st.sidebar.radio(
    "Initialize inputs from",
    ["Median machine", "A real record", "A real event"],
    help="Choose what the sliders start at. You can still tweak everything below.",
)

if "seed" not in st.session_state:
    st.session_state.seed = {f: float(df[f].median()) for f in config.RAW_FEATURES}
    st.session_state.seed_date = df[config.DATE_COL].median().date()

if seed_mode == "Median machine":
    if st.sidebar.button("Reset to median values"):
        st.session_state.seed = {f: float(df[f].median()) for f in config.RAW_FEATURES}
        st.session_state.seed_date = df[config.DATE_COL].median().date()
else:
    pool = df[df[config.TARGET_COL] == 1] if seed_mode == "A real event" else df
    label = "event record" if seed_mode == "A real event" else "record"
    idx_options = pool.index.tolist()
    if st.sidebar.button(f"🎲 Load a random {label}"):
        row = pool.loc[np.random.choice(idx_options)]
        st.session_state.seed = {f: float(row[f]) for f in config.RAW_FEATURES}
        st.session_state.seed_date = row[config.DATE_COL].date()
        st.session_state.seed_machine = row[config.MACHINE_COL]
    st.sidebar.caption(
        f"Loaded from: **{st.session_state.get('seed_machine', '—')}**"
        if "seed_machine" in st.session_state else "Click the button to load a record."
    )

# --------------------------------------------------------------------------- #
# Input controls
# --------------------------------------------------------------------------- #
st.subheader("Daily telemetry input")
event_date = st.date_input(
    "Record date",
    value=st.session_state.seed_date,
    min_value=df[config.DATE_COL].min().date(),
    max_value=df[config.DATE_COL].max().date(),
    help="Drives the temporal features (day of week / day of month / month).",
)

cols = st.columns(3)
values: dict[str, float] = {}
for i, feat in enumerate(config.RAW_FEATURES):
    col = cols[i % 3]
    fmax = float(feat_stats.loc[feat, "99%"])
    fmax = max(fmax, st.session_state.seed[feat], 1.0)
    values[feat] = col.slider(
        feat,
        min_value=0.0,
        max_value=round(fmax * 1.5, 2),
        value=float(min(st.session_state.seed[feat], fmax * 1.5)),
        help=f"median={feat_stats.loc[feat, '50%']:.0f} · "
             f"99th pct={feat_stats.loc[feat, '99%']:.0f} · "
             f"max={feat_stats.loc[feat, 'max']:.0f}",
    )

threshold = st.slider(
    "Decision threshold",
    min_value=0.0, max_value=0.5, value=float(model.threshold), step=0.001,
    format="%.3f",
    help=f"An event is flagged when probability ≥ threshold. "
         f"Tuned default = {model.threshold}.",
)

# --------------------------------------------------------------------------- #
# Build the feature row and predict
# --------------------------------------------------------------------------- #
ts = pd.Timestamp(event_date)
row = {**values,
       "day_of_week": ts.dayofweek,
       "day_of_month": ts.day,
       "month": ts.month}
base = pd.DataFrame([row])[config.BASE_FEATURES]

prob = float(model.predict_proba(base)[0])
anomaly = float(model.anomaly_score(base)[0])
flagged = prob >= threshold
label, color = config.risk_band(prob)

# Anomaly percentile relative to the dataset (computed once, cached).
@st.cache_data(show_spinner=False)
def _anomaly_distribution() -> np.ndarray:
    return model.anomaly_score(df[config.BASE_FEATURES])

anom_dist = _anomaly_distribution()
anom_pct = float((anom_dist < anomaly).mean())

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Model assessment")

res_left, res_right = st.columns([1.2, 1])
with res_left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "valueformat": ".2f"},
        title={"text": "Event probability"},
        gauge={
            "axis": {"range": [0, max(prob * 100 * 1.3, 10)]},
            "bar": {"color": color},
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    gauge.update_layout(height=280, margin=dict(t=50, b=10))
    st.plotly_chart(gauge, use_container_width=True)

with res_right:
    st.markdown(ui.risk_badge(prob), unsafe_allow_html=True)
    if flagged:
        st.error(f"🚨 **EVENT FLAGGED** — probability {prob:.2%} ≥ threshold {threshold:.3f}")
    else:
        st.success(f"✅ No event flagged — probability {prob:.2%} < threshold {threshold:.3f}")

    st.metric("Anomaly score (Isolation Forest)", f"{anomaly:.3f}",
              help="Higher = more unusual relative to normal machine behavior.")
    st.caption(f"More anomalous than **{anom_pct:.1%}** of all records in the dataset.")
    st.metric("Event probability", f"{prob:.3%}")

# --------------------------------------------------------------------------- #
# How inputs compare to the fleet
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("How these inputs compare to the fleet")
st.caption("Percentile of each input value within the full dataset. Extreme values push risk up.")

pct_rows = []
for feat in config.RAW_FEATURES:
    pctl = float((df[feat] < values[feat]).mean())
    pct_rows.append({"feature": feat, "value": values[feat], "percentile": pctl})
pct_df = pd.DataFrame(pct_rows)

fig = go.Figure(go.Bar(
    x=pct_df["percentile"] * 100, y=pct_df["feature"], orientation="h",
    marker_color=["#e74c3c" if p > 0.95 else "#3498db" for p in pct_df["percentile"]],
    text=[f"{p:.0%}" for p in pct_df["percentile"]], textposition="auto",
))
fig.update_layout(
    xaxis_title="Percentile within dataset", yaxis_title="",
    height=380, margin=dict(t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Bars in red are above the 95th percentile — unusually high for the fleet.")
