"""Smart Production Line Monitoring and Quality Prediction Using Industrial IoT.

Run with:  streamlit run app.py
"""
from datetime import timedelta
from pathlib import Path

import streamlit as st

from src import config as C
from src import data
from views import about, analytics, anomalies, prediction, process, production, quality

st.set_page_config(page_title="Smart Production Line Monitoring", page_icon="🏭", layout="wide")

DEFAULT_CSV = Path(__file__).parent / "data" / "continuous_factory_process.csv"

PAGES = {
    "1 · Production monitoring": production,
    "2 · Process parameters": process,
    "3 · Quality monitoring": quality,
    "4 · Abnormality detection": anomalies,
    "5 · Production analytics": analytics,
    "6 · Quality prediction (experimental)": prediction,
    "About & limitations": about,
}

st.sidebar.title("🏭 Smart Production Line")
st.sidebar.caption("Industrial IoT monitoring · multi-stage continuous-flow process")

upload = st.sidebar.file_uploader("Use a different CSV (same layout)", type="csv")
try:
    df_full = data.load_data(upload if upload is not None else DEFAULT_CSV)
except Exception as exc:  # unreadable file
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

missing = [c for c in (C.TIME_COL, C.actual_col(1, 0), C.setpoint_col(1, 0), C.actual_col(2, 0)) if c not in df_full.columns]
if missing:
    st.error(f"The CSV does not have the expected columns (missing {missing}).")
    st.stop()

page = st.sidebar.radio("Module", list(PAGES))

t0, t1 = df_full[C.TIME_COL].min().to_pydatetime(), df_full[C.TIME_COL].max().to_pydatetime()
lo, hi = st.sidebar.slider("Time window", min_value=t0, max_value=t1, value=(t0, t1),
                           step=timedelta(seconds=30), format="HH:mm:ss")
df = df_full[(df_full[C.TIME_COL] >= lo) & (df_full[C.TIME_COL] <= hi)].reset_index(drop=True)
if df.empty:
    st.warning("No records in the selected time window.")
    st.stop()
st.sidebar.caption(f"{len(df):,} of {len(df_full):,} records selected")

PAGES[page].render(df_full, df)
