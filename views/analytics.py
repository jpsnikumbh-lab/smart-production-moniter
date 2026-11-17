"""Module 5 - Production analytics: compare stages, find associated drivers, build a report."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import anomaly, config as C, data, quality, reporting


def stage_comparison(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    s1, s2 = quality.stage_summary(df, 1, tol), quality.stage_summary(df, 2, tol)
    rows = []
    for i in range(C.N_MEAS):
        f1, f2 = quality.measurement_frame(df, 1, i), quality.measurement_frame(df, 2, i)
        both = f1["valid"] & f2["valid"]
        corr = f1.loc[both, "actual"].corr(f2.loc[both, "actual"]) if both.sum() > 30 else np.nan
        rows.append({"measurement": f"M{i}",
                     "S1_coverage_pct": s1.loc[i, "coverage_pct"], "S2_coverage_pct": s2.loc[i, "coverage_pct"],
                     "S1_mae_pct": s1.loc[i, "mae_pct_of_setpoint"], "S2_mae_pct": s2.loc[i, "mae_pct_of_setpoint"],
                     "S1_std_mm": s1.loc[i, "std_error_mm"], "S2_std_mm": s2.loc[i, "std_error_mm"],
                     "corr_S1_vs_S2": corr})
    return pd.DataFrame(rows)


def driver_table(df, stage, i, method):
    f = quality.measurement_frame(df, stage, i)
    feats = [c for c in data.feature_columns(df, stage) if df[c].nunique() > 1]
    sub = df.loc[f["valid"], feats]
    if len(sub) < 100:
        return pd.DataFrame(columns=["variable", "correlation"])
    r = sub.corrwith(f.loc[f["valid"], "actual"], method=method).dropna()
    return pd.DataFrame({"variable": r.index, "correlation": r.values}).assign(
        abs_corr=lambda t: t["correlation"].abs()).sort_values("abs_corr", ascending=False)


def _machine_of(col):
    for p in ["Machine1", "Machine2", "Machine3", "Machine4", "Machine5", "FirstStage", "AmbientConditions"]:
        if col.startswith(p):
            return {"FirstStage": "Combiner", "AmbientConditions": "Ambient"}.get(p, p)
    return "Other"


def render(df_full: pd.DataFrame, df: pd.DataFrame):
    st.header("5 · Production Analytics")
    tol = st.sidebar.slider("Analytics tolerance (% of setpoint)", 1.0, 30.0, C.DEFAULT_TOL_PCT, 0.5, key="an_tol")

    st.subheader("Stage 1 vs Stage 2")
    cmp_ = stage_comparison(df, tol)
    st.dataframe(cmp_.round(3), hide_index=True, width="stretch")
    long = cmp_.melt(id_vars="measurement", value_vars=["S1_mae_pct", "S2_mae_pct"], var_name="stage", value_name="mae_pct")
    long["stage"] = long["stage"].map({"S1_mae_pct": "Stage 1", "S2_mae_pct": "Stage 2"})
    fig = px.bar(long, x="measurement", y="mae_pct", color="stage", barmode="group",
                 labels={"mae_pct": "Mean |error| (% of setpoint)"})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, title=""))
    st.plotly_chart(fig, width="stretch")
    worse = int((cmp_["S2_std_mm"] > cmp_["S1_std_mm"]).sum())
    st.caption(f"Stage 2 error spread (std, mm) is larger than stage 1 for {worse} of 15 measurements in this window. "
               "The dataset notes say stage-2 data is noisier; this table lets you check that claim per measurement.")

    st.subheader("Which upstream variables move with a measurement?")
    c1, c2, c3, c4 = st.columns(4)
    stage = c1.radio("Stage", [1, 2], horizontal=True, key="drv_stage")
    i = c2.selectbox("Measurement", range(C.N_MEAS), format_func=lambda k: f"M{k}", key="drv_i")
    method = c3.selectbox("Correlation", ["pearson", "spearman"])
    topn = c4.slider("Top N", 5, 25, 10)
    drv = driver_table(df, stage, i, method)
    if drv.empty:
        st.warning("Not enough valid readings for this measurement in the selected window.")
    else:
        top = drv.head(topn)
        figd = px.bar(top, x="correlation", y="variable", orientation="h", range_x=[-1, 1])
        figd.update_layout(height=max(300, 26 * len(top)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figd, width="stretch")
        by = drv.assign(machine=drv["variable"].map(_machine_of)).groupby("machine")["abs_corr"].mean().sort_values(ascending=False)
        st.caption("Average |correlation| by source (a rough guide to where the association sits):")
        st.bar_chart(by)
        st.warning("Correlation is association, not cause. A strong value can come from both variables drifting with "
                   "time. Use it to choose what to investigate, not to conclude what to adjust.")

    st.subheader("Generate report")
    st.caption("Builds a standalone HTML report for the selected time window using the settings on this page.")
    if st.button("Build report"):
        health = data.data_health(df_full)
        s1, s2 = quality.stage_summary(df, 1, tol), quality.stage_summary(df, 2, tol)
        html = reporting.build_report_html(df, health, tol, s1, s2, cmp_)
        st.session_state["report_html"] = html
    if "report_html" in st.session_state:
        st.download_button("Download report (HTML)", st.session_state["report_html"].encode(),
                           "production_quality_report.html", "text/html")
        st.download_button("Download stage comparison (CSV)", cmp_.to_csv(index=False).encode(), "stage_comparison.csv", "text/csv")
