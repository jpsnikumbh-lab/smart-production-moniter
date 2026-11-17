"""Module 4 - Abnormality detection using statistical thresholds."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src import anomaly, config as C, data, quality

SCOPES = ["Process variables (machines, combiner, ambient)", "Stage 1 output measurements", "Stage 2 output measurements"]


def _series_for(df, scope, name):
    if scope.startswith("Process"):
        return df[name]
    stage = 1 if "Stage 1" in scope else 2
    k = int(name.replace("M", ""))
    return quality.measurement_frame(df, stage, k)["actual_valid"]


def _names(df, scope):
    if scope.startswith("Process"):
        cols = data.feature_columns(df, 2)
        return [c for c in cols if df[c].nunique() > 1]
    return [f"M{i}" for i in range(C.N_MEAS)]


def render(df_full: pd.DataFrame, df: pd.DataFrame):
    st.header("4 · Abnormality Detection")
    st.info("Flags are **statistical** (a value is unusual compared with that variable's own history). They show where "
            "to look; they do not prove a defective product. No labelled defects exist in this dataset, so detection "
            "accuracy cannot be measured and is not claimed.")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    scope = c1.selectbox("Scope", SCOPES)
    names = _names(df, scope)
    var = c2.selectbox("Variable", names)
    method = c3.selectbox("Method", anomaly.METHODS)
    default_thr = 1.5 if method == "IQR fences" else 4.0
    thr = c4.number_input("Threshold", 0.5, 10.0, default_thr, 0.5,
                          help="z-score limit (σ) or IQR multiplier k.")
    window = 300
    if method == "Rolling z-score":
        window = st.slider("Rolling window (seconds)", 30, 900, 300, 30)

    s = _series_for(df, scope, var)
    flags = anomaly.detect(s, method, thr, window).fillna(False)
    n_valid = int(s.notna().sum())
    m = st.columns(3)
    m[0].metric("Flagged points", int(flags.sum()))
    m[1].metric("Share of valid points", f"{100 * flags.sum() / max(n_valid, 1):.2f}%")
    events = anomaly.events_from_flags(df[C.TIME_COL], s, flags)
    m[2].metric("Events (merged)", len(events))

    d = data.downsample(pd.DataFrame({"t": df[C.TIME_COL], "v": s}), 3500)
    fl = pd.DataFrame({"t": df[C.TIME_COL][flags], "v": s[flags]})
    if len(fl) > 3000:
        fl = fl.iloc[:: int(np.ceil(len(fl) / 3000))]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=d["t"], y=d["v"], mode="lines", name="value", line=dict(width=1)))
    fig.add_trace(go.Scattergl(x=fl["t"], y=fl["v"], mode="markers", name="flagged",
                               marker=dict(color="red", size=6)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Detected events")
    if events.empty:
        st.success("No anomalies with these settings.")
    else:
        st.dataframe(events.round(3), hide_index=True, width="stretch")
        st.download_button("Download events (CSV)", events.to_csv(index=False).encode(), "anomaly_events.csv", "text/csv")

    st.subheader("Scan all variables in this scope")
    st.caption("Counts flagged points per variable with the method and threshold chosen above.")
    if st.checkbox("Run scan", value=True):
        rows = []
        for nm in names:
            sv = _series_for(df, scope, nm)
            fl_all = anomaly.detect(sv, method, thr, window).fillna(False)
            nv = int(sv.notna().sum())
            if nv:
                rows.append({"variable": nm, "flagged": int(fl_all.sum()), "pct_flagged": 100 * fl_all.sum() / nv})
        scan = pd.DataFrame(rows).sort_values("flagged", ascending=False)
        top = scan.head(15)
        figb = px.bar(top, x="flagged", y="variable", orientation="h", hover_data=["pct_flagged"])
        figb.update_layout(height=max(300, 24 * len(top)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figb, width="stretch")
        st.download_button("Download scan (CSV)", scan.to_csv(index=False).encode(), "anomaly_scan.csv", "text/csv")
