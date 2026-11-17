"""Module 6 (extension) - quality prediction, with validation reported honestly."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C, modeling


@st.cache_data(show_spinner="Running forward-chaining validation for all 15 targets (about a minute) ...")
def _all(df, stage, model, n_est):
    return modeling.all_targets_summary(df, stage, model, n_est, 4)


def render(df_full: pd.DataFrame, df: pd.DataFrame):
    st.header("6 · Quality Prediction (experimental)")
    st.warning("This page tests whether a measurement can be predicted from upstream process variables. It reports "
               "results on **later, unseen data** and compares them with a naive baseline. A model is only called "
               "useful if it beats that baseline. Uses the **full run**, not the sidebar time window, so the "
               "chronological split is meaningful.")

    c1, c2, c3, c4 = st.columns(4)
    stage = c1.radio("Stage", [1, 2], horizontal=True, key="pr_stage")
    i = c2.selectbox("Target measurement", range(C.N_MEAS), format_func=lambda k: f"M{k}", key="pr_i")
    model = c3.selectbox("Model", modeling.MODELS)
    frac = c4.slider("Train share (first part of run)", 0.5, 0.85, 0.7, 0.05)
    n_est = 100
    if model == "Random Forest":
        n_est = st.slider("Trees", 20, 300, 100, 20)

    res = modeling.chrono_evaluate(df_full, stage, i, model, frac, n_est)
    if not res["ok"]:
        st.error(res["reason"])
    else:
        m = st.columns(5)
        m[0].metric("Train / test rows", f"{res['n_train']:,} / {res['n_test']:,}")
        m[1].metric("Test R²", f"{res['r2']:.2f}")
        m[2].metric("Test MAE (mm)", f"{res['mae']:.3f}")
        m[3].metric("Baseline MAE (mm)", f"{res['baseline_mae']:.3f}")
        m[4].metric("MAE improvement", f"{res['improvement_pct']:+.1f}%")
        if res["verdict"].startswith("Beats"):
            st.success(res["verdict"] + ". Confirm with the forward-chaining table below before relying on it.")
        else:
            st.error(res["verdict"] + ".")
        st.caption("Baseline = always predict the average of the training period. R² below 0 means worse than that.")

        t = res["test_time"]
        step = max(1, len(t) // 2500)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=t.iloc[::step], y=res["y_test"].iloc[::step], mode="markers", marker=dict(size=3), name="Actual"))
        fig.add_trace(go.Scattergl(x=t.iloc[::step], y=res["pred"].iloc[::step], mode="markers", marker=dict(size=3), name="Predicted"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="mm", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption("Held-out later part of the run.")
        st.subheader("Most influential inputs (model-specific, not causal)")
        st.bar_chart(res["importance"])

    st.subheader("Forward-chaining validation across all 15 targets")
    st.caption("Four expanding-window folds in time order with a 60 s gap. 'Beats baseline' needs R² > 0 and ≥ 5% lower MAE.")
    if st.button("Run validation for all targets"):
        st.session_state["pred_summary"] = _all(df_full, stage, model, 30 if model == "Random Forest" else 100)
    if "pred_summary" in st.session_state:
        st.dataframe(st.session_state["pred_summary"].round(3), hide_index=True, width="stretch")
        st.download_button("Download validation table (CSV)", st.session_state["pred_summary"].to_csv(index=False).encode(),
                           "prediction_validation.csv", "text/csv")
