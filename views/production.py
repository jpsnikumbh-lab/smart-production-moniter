"""Module 1 - Production monitoring: records, processing stages, data health."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import data

FLOW = """
digraph G {
  rankdir=LR; nodesep=0.3;
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#4a6fa5", fontname="Helvetica"];
  edge [color="#4a6fa5"];
  M1 [label="Machine 1"]; M2 [label="Machine 2"]; M3 [label="Machine 3"];
  CB [label="Combiner", fillcolor="#fff4d6"];
  S1 [label="STAGE 1 output\\n15 measurements (primary)", fillcolor="#e3f4e6"];
  M4 [label="Machine 4"]; M5 [label="Machine 5"];
  S2 [label="STAGE 2 output\\n15 measurements (secondary)", fillcolor="#e3f4e6"];
  {rank=same; M1; M2; M3}
  M1 -> CB; M2 -> CB; M3 -> CB; CB -> S1; S1 -> M4; M4 -> M5; M5 -> S2;
}
"""


def render(df_full: pd.DataFrame, df: pd.DataFrame):
    st.header("1 · Production Monitoring")
    st.caption("Overview of the production records and the two processing stages. "
               "All values on this page are measured/recorded data; nothing here is predicted.")
    h = data.data_health(df_full)

    c = st.columns(5)
    c[0].metric("Records (full file)", f"{h['records']:,}")
    c[1].metric("Variables", h["variables"])
    c[2].metric("Records in window", f"{len(df):,}")
    span = (h["end"] - h["start"])
    c[3].metric("Run length", str(span).replace("0 days ", ""))
    c[4].metric("Sampling", "1 Hz")

    st.subheader("Process flow")
    st.graphviz_chart(FLOW, width="stretch")
    st.caption("Machines 1-3 run in parallel and feed a combiner; the combined flow is measured at 15 locations "
               "(stage 1). It then passes through machines 4 and 5 in series and is measured again (stage 2).")

    st.subheader("Sensor availability over time")
    st.caption("Share of the 15 output measurements that have a usable (non-zero) reading, 60-second rolling mean. "
               "Drops show dropouts or idle periods.")
    fig = go.Figure()
    for stage, name in [(1, "Stage 1"), (2, "Stage 2")]:
        acts = df[[C.actual_col(stage, i) for i in range(C.N_MEAS)]]
        share = (acts > C.EPS).mean(axis=1).rolling(60, min_periods=1).mean() * 100
        d = data.downsample(pd.DataFrame({"t": df[C.TIME_COL], "v": share}))
        fig.add_trace(go.Scatter(x=d["t"], y=d["v"], name=name, mode="lines"))
    fig.update_layout(height=300, yaxis_title="% measurements with a reading", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Variable groups")
    groups = data.group_columns(df_full)
    rows = [{"Group": g, "Columns": f"{C.COLUMN_GROUPS[g][0]}-{C.COLUMN_GROUPS[g][1]}", "Variables": len(cols),
             "Example": cols[0]} for g, cols in groups.items()]
    rows += [{"Group": "Stage 1 output (actual + setpoint)", "Columns": "42-71", "Variables": 30,
              "Example": C.actual_col(1, 0)},
             {"Group": "Stage 2 output (actual + setpoint)", "Columns": "86-115", "Variables": 30,
              "Example": C.actual_col(2, 0)}]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Data health checks")
    a, b, c2, d = st.columns(4)
    a.metric("Missing cells", h["missing_cells"])
    b.metric("Duplicate timestamps", h["duplicate_timestamps"])
    c2.metric("Non-1 s time steps", h["irregular_steps"])
    d.metric("Stage-1 idle rows", h["stage1_idle_rows"])
    if h["stage1_idle_rows"]:
        st.info(f"All 15 stage-1 setpoints are 0 in {h['stage1_idle_rows']} rows in total (first at "
                f"{h['stage1_idle_start']}, last at {h['stage1_idle_end']}). We treat these rows as idle/changeover "
                "time and exclude them from quality calculations.")
    with st.expander("Constant columns"):
        st.write(f"{len(h['constant_columns'])} columns never change (all stage-2 setpoints are constant).")
        st.write(h["constant_columns"])

    st.subheader("Record explorer")
    gname = st.selectbox("Variable group", list(groups) + ["Stage 1 output", "Stage 2 output"])
    if gname == "Stage 1 output":
        cols = [c for i in range(C.N_MEAS) for c in (C.actual_col(1, i), C.setpoint_col(1, i))]
    elif gname == "Stage 2 output":
        cols = [c for i in range(C.N_MEAS) for c in (C.actual_col(2, i), C.setpoint_col(2, i))]
    else:
        cols = groups[gname]
    n = st.slider("Rows to show", 10, 500, 100, step=10)
    start = st.number_input("Start at row (within window)", 0, max(0, len(df) - 1), 0, step=100)
    view = df[[C.TIME_COL] + cols].iloc[int(start): int(start) + n]
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button("Download this group for the selected window (CSV)",
                       df[[C.TIME_COL] + cols].to_csv(index=False).encode(), f"{gname.replace(' ', '_')}.csv", "text/csv")
