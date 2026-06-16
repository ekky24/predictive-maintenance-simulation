"""Data Explorer — distributions, event timeline, and per-machine drill-down."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from pdm import config
from pdm import streamlit_utils as ui

ui.configure_page("Data Explorer")
st.title("📊 Data Explorer")
st.caption("Understand the telemetry, the class imbalance, and how events are distributed.")

df = ui.load_data()

# --------------------------------------------------------------------------- #
# Top-line counts
# --------------------------------------------------------------------------- #
events = df[df[config.TARGET_COL] == 1]
ui.metric_row([
    ("Records", f"{len(df):,}", None),
    ("Machines", f"{df[config.MACHINE_COL].nunique():,}", None),
    ("Events", f"{len(events):,}", None),
    ("Event rate", f"{df[config.TARGET_COL].mean():.3%}", None),
])

tab_overview, tab_features, tab_timeline, tab_machine = st.tabs(
    ["Class balance", "Feature distributions", "Event timeline", "Machine drill-down"]
)

# --------------------------------------------------------------------------- #
# Class balance
# --------------------------------------------------------------------------- #
with tab_overview:
    st.subheader("Extreme class imbalance")
    st.markdown(
        "Events are the rare positive class. This is *the* central challenge: a "
        "model that always predicts **No Event** would be ~99.9% accurate yet "
        "completely useless."
    )
    counts = (
        df[config.TARGET_COL]
        .map({0: "No Event", 1: "Event"})
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            counts, x="class", y="count", color="class", text="count",
            color_discrete_map={"No Event": "#3498db", "Event": "#e74c3c"},
            log_y=True, title="Class distribution (log scale)",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(counts, hide_index=True, use_container_width=True)
        rate = df[config.TARGET_COL].mean()
        st.metric("Imbalance ratio", f"1 : {round(1 / rate):,}")

# --------------------------------------------------------------------------- #
# Feature distributions
# --------------------------------------------------------------------------- #
with tab_features:
    st.subheader("Feature distributions: events vs. non-events")
    st.markdown(
        "Compare how a feature behaves on event days versus normal days. "
        "Differences here are what the model exploits."
    )
    feature = st.selectbox("Feature", config.RAW_FEATURES, index=5)
    log_x = st.checkbox("Log-scale the x-axis", value=True,
                        help="Useful — several features are heavily right-skewed.")

    plot_df = df[[feature, config.TARGET_COL]].copy()
    plot_df["class"] = plot_df[config.TARGET_COL].map({0: "No Event", 1: "Event"})
    if log_x:
        plot_df = plot_df[plot_df[feature] > 0]

    fig = px.histogram(
        plot_df, x=feature, color="class", barmode="overlay",
        histnorm="probability density", nbins=60, log_x=log_x,
        color_discrete_map={"No Event": "#3498db", "Event": "#e74c3c"},
        title=f"Distribution of {feature} by class"
        + (" (positive values, log x)" if log_x else ""),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Summary statistics by class**")
    st.dataframe(
        df.groupby(config.TARGET_COL)[feature]
        .describe()
        .rename(index={0: "No Event", 1: "Event"}),
        use_container_width=True,
    )

# --------------------------------------------------------------------------- #
# Event timeline
# --------------------------------------------------------------------------- #
with tab_timeline:
    st.subheader("When do events happen?")
    freq = st.radio("Aggregate by", ["Day", "Week", "Month"], horizontal=True, index=1)
    rule = {"Day": "D", "Week": "W", "Month": "MS"}[freq]

    daily = (
        df.set_index(config.DATE_COL)[config.TARGET_COL]
        .resample(rule)
        .agg(["sum", "count"])
        .rename(columns={"sum": "events", "count": "records"})
        .reset_index()
    )
    fig = px.bar(
        daily, x=config.DATE_COL, y="events",
        title=f"Machine events per {freq.lower()}",
        labels={"events": "Number of events", config.DATE_COL: "Date"},
    )
    fig.update_traces(marker_color="#e74c3c")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        dow = (
            events.assign(dow=events[config.DATE_COL].dt.day_name())
            .groupby("dow").size()
            .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"])
            .reset_index(name="events")
        )
        fig = px.bar(dow, x="dow", y="events", title="Events by day of week")
        fig.update_traces(marker_color="#9b59b6")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        month = (
            events.assign(m=events[config.DATE_COL].dt.month_name())
            .groupby("m").size().reset_index(name="events")
        )
        fig = px.bar(month, x="m", y="events", title="Events by month")
        fig.update_traces(marker_color="#1abc9c")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Machine drill-down
# --------------------------------------------------------------------------- #
with tab_machine:
    st.subheader("Per-machine drill-down")
    machines_with_events = sorted(events[config.MACHINE_COL].unique())
    st.caption(
        f"{len(machines_with_events)} of {df[config.MACHINE_COL].nunique():,} "
        "machines recorded at least one event."
    )
    only_event_machines = st.checkbox("Only show machines that had an event", value=True)
    options = machines_with_events if only_event_machines else sorted(df[config.MACHINE_COL].unique())
    machine = st.selectbox("Machine", options)

    m = df[df[config.MACHINE_COL] == machine].sort_values(config.DATE_COL)
    ui.metric_row([
        ("Records", f"{len(m):,}", None),
        ("Events", f"{int(m[config.TARGET_COL].sum())}", None),
        ("First seen", str(m[config.DATE_COL].min().date()), None),
        ("Last seen", str(m[config.DATE_COL].max().date()), None),
    ])

    feature = st.selectbox("Telemetry feature to plot", config.RAW_FEATURES, index=0, key="m_feat")
    fig = px.line(m, x=config.DATE_COL, y=feature, title=f"{feature} over time — {machine}")
    ev = m[m[config.TARGET_COL] == 1]
    if not ev.empty:
        fig.add_scatter(
            x=ev[config.DATE_COL], y=ev[feature], mode="markers",
            marker=dict(color="#e74c3c", size=11, symbol="x"), name="Event",
        )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw records for this machine"):
        st.dataframe(m, hide_index=True, use_container_width=True)
