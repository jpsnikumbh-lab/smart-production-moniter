"""Module 2 - Process parameter monitoring: trends and distributions of measured variables."""
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src import config as C
from src import data


def render(df_full: pd.DataFrame, df: pd.DataFrame):
    st.header("2 · Process Parameter Monitoring")
    st.caption("Trends of the actual manufacturing measurements recorded by the sensors. "
               "Naming convention: C = controlled variable, U = uncontrolled variable.")
    groups = data.group_columns(df_full)
    g = st.selectbox("Variable group", list(groups))
    cols = groups[g]
    default = cols[: min(3, len(cols))]
    chosen = st.multiselect("Variables to plot (max 6)", cols, default=default, max_selections=6)
    smooth = st.slider("Rolling-mean overlay (seconds, 1 = off)", 1, 300, 1)
    if not chosen:
        st.info("Select at least one variable.")
        return

    d = data.downsample(df[[C.TIME_COL] + chosen], 3000)
    fig = make_subplots(rows=len(chosen), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=[c.split(".", 1)[-1] if "." in c else c for c in chosen])
    for k, col in enumerate(chosen, start=1):
        fig.add_trace(go.Scattergl(x=d[C.TIME_COL], y=d[col], mode="lines", name=col, line=dict(width=1)), row=k, col=1)
        if smooth > 1:
            rm = df[col].rolling(smooth, min_periods=1).mean().iloc[:: max(1, len(df) // 3000)]
            fig.add_trace(go.Scattergl(x=d[C.TIME_COL], y=rm.iloc[: len(d)], mode="lines",
                                       name=f"{col} (mean {smooth}s)", line=dict(width=2, color="black")), row=k, col=1)
    fig.update_layout(height=190 * len(chosen) + 80, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Summary statistics (selected window)")
    stats = df[chosen].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.insert(0, "type", [data.variable_kind(c) for c in stats.index])
    stats["cv_pct"] = (stats["std"] / stats["mean"].abs() * 100).where(stats["mean"].abs() > 1e-9)
    st.dataframe(stats.round(3), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Distribution")
        v = st.selectbox("Variable", chosen, key="dist_var")
        st.plotly_chart(px.histogram(df, x=v, nbins=60).update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10)),
                        width="stretch")
    with right:
        st.subheader("Correlation within group")
        if len(cols) >= 2:
            corr = df[cols].corr()
            fig2 = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
            fig2.update_xaxes(showticklabels=False)
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, width="stretch")
            st.caption("Hover for variable names. Correlation shows co-movement, not cause and effect.")
