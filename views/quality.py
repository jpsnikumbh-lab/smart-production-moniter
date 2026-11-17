"""Module 3 - Quality monitoring: measurement vs setpoint, error and a DERIVED tolerance label."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import data, quality


def render(df_full: pd.DataFrame, df: pd.DataFrame):
    st.header("3 · Quality Monitoring")
    st.info("**Measured vs derived.** Actual and setpoint values are *measured data*. The dataset has **no pass/fail "
            "label**, so 'within / outside tolerance' below is a *derived* label from a rule you choose "
            "(|actual − setpoint| ≤ tolerance % of setpoint). It is not ground-truth product quality.")

    c1, c2, c3 = st.columns(3)
    stage = c1.radio("Stage", [1, 2], format_func=lambda s: f"Stage {s} " + ("(primary)" if s == 1 else "(secondary, noisier)"),
                     horizontal=True)
    i = c2.selectbox("Measurement", range(C.N_MEAS), format_func=lambda k: f"Measurement {k}")
    tol = c3.slider("Tolerance (% of setpoint)", 1.0, 30.0, C.DEFAULT_TOL_PCT, 0.5)

    f = quality.measurement_frame(df, stage, i)
    lab = quality.classify(f, tol)
    v = f[f["valid"]]
    if v.empty:
        st.warning("No valid readings for this measurement in the selected window.")
        return

    active = int((f["status"] != "idle").sum())
    m = st.columns(5)
    m[0].metric("Setpoint (mm)", f"{v['setpoint'].median():.2f}")
    m[1].metric("Mean actual (mm)", f"{v['actual'].mean():.2f}")
    m[2].metric("Bias (mm)", f"{v['error'].mean():+.2f}")
    m[3].metric("Valid readings", f"{100 * len(v) / max(active, 1):.1f}%")
    m[4].metric("Within tolerance*", f"{100 * (v['rel_error_pct'].abs() <= tol).mean():.1f}%")
    st.caption("*derived label, share of valid readings.")
    if len(v) / max(active, 1) < 0.5:
        st.warning("Fewer than half of the readings for this measurement are non-zero - treat conclusions with caution.")
    if abs(v["error"].mean()) > 0.2 * v["setpoint"].median():
        st.warning("The average error is more than 20% of the setpoint: a persistent offset between setpoint and "
                   "actual, not random noise. Check whether the setpoint really corresponds to this sensor.")

    d = data.downsample(f[f["valid"]], 4000)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=d[C.TIME_COL], y=d["actual"], mode="markers", marker=dict(size=3), name="Actual"))
    sp = data.downsample(f, 4000)
    sp_line = sp["setpoint"].where(sp["setpoint"] > C.EPS)
    fig.add_trace(go.Scatter(x=sp[C.TIME_COL], y=sp_line, mode="lines", name="Setpoint", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=sp[C.TIME_COL], y=sp_line * (1 + tol / 100), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sp[C.TIME_COL], y=sp_line * (1 - tol / 100), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(0,160,80,0.15)", name=f"±{tol:g}% band"))
    fig.update_layout(height=380, yaxis_title="mm", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Error distribution (actual − setpoint)")
        st.plotly_chart(px.histogram(v, x="error", nbins=60).update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10)),
                        width="stretch")
    with right:
        st.subheader("Derived classification over time")
        binsz = st.selectbox("Bin size", ["1min", "5min", "15min"], index=1)
        t = pd.DataFrame({"bin": pd.to_datetime(f[C.TIME_COL]).dt.floor(binsz), "label": lab})
        cnt = t.groupby(["bin", "label"]).size().reset_index(name="rows")
        figb = px.bar(cnt, x="bin", y="rows", color="label", color_discrete_map={
            quality.WITHIN: "#2e9e5b", quality.OUTSIDE: "#d9534f", quality.NO_READING: "#9aa0a6", quality.IDLE: "#f0ad4e"})
        figb.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25, title=""))
        st.plotly_chart(figb, width="stretch")

    st.subheader(f"All 15 measurements - Stage {stage}")
    summ = quality.stage_summary(df, stage, tol)
    st.dataframe(summ.round(3), hide_index=True, width="stretch", column_config={
        "coverage_pct": st.column_config.ProgressColumn("coverage_pct", min_value=0, max_value=100, format="%.0f%%"),
        "within_tol_pct": st.column_config.ProgressColumn("within_tol_pct (derived)", min_value=0, max_value=100, format="%.0f%%"),
    })
    st.caption("coverage_pct = share of non-idle rows with a non-zero reading. Measurements with low coverage "
               "(e.g. M5, M11) rest on few readings.")
    fig3 = px.bar(summ, x="measurement", y="mae_pct_of_setpoint", labels={"mae_pct_of_setpoint": "Mean |error| (% of setpoint)"})
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, width="stretch")
    st.download_button("Download table (CSV)", summ.to_csv(index=False).encode(), f"stage{stage}_quality_summary.csv", "text/csv")
