"""Data loading, schema helpers and basic data-health checks."""
import math

import numpy as np
import pandas as pd
import streamlit as st

from . import config as C


@st.cache_data(show_spinner="Loading dataset ...")
def load_data(source) -> pd.DataFrame:
    """Read the CSV (path or uploaded buffer), parse timestamps and sort by time."""
    df = pd.read_csv(source)
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    return df.sort_values(C.TIME_COL, kind="stable").reset_index(drop=True)


def columns_in_range(df: pd.DataFrame, first: int, last: int) -> list:
    return list(df.columns[first:last + 1])


def group_columns(df: pd.DataFrame) -> dict:
    return {name: columns_in_range(df, a, b) for name, (a, b) in C.COLUMN_GROUPS.items()}


def variable_kind(col: str) -> str:
    """Controlled / uncontrolled / raw-material-or-state, from the naming convention."""
    if ".C." in col:
        return "Controlled (C)"
    if ".U." in col:
        return "Uncontrolled (U)"
    return "Raw material / state"


def feature_columns(df: pd.DataFrame, stage: int) -> list:
    """Upstream columns that can legitimately be used to explain a stage's output."""
    cols = columns_in_range(df, *C.STAGE1_FEATURE_RANGE)
    if stage == 2:
        for a, b in C.STAGE2_EXTRA_RANGES:
            cols += columns_in_range(df, a, b)
    return cols


def downsample(df: pd.DataFrame, max_points: int = 3000) -> pd.DataFrame:
    step = max(1, math.ceil(len(df) / max_points))
    return df.iloc[::step]


def data_health(df: pd.DataFrame) -> dict:
    """Basic integrity checks reported on the Production Monitoring page."""
    dt = df[C.TIME_COL].diff().dt.total_seconds().dropna()
    num = df.drop(columns=[C.TIME_COL])
    sp1 = df[[C.setpoint_col(1, i) for i in range(C.N_MEAS)]]
    idle = (sp1 <= C.EPS).all(axis=1)
    return {
        "records": len(df),
        "variables": df.shape[1],
        "start": df[C.TIME_COL].min(),
        "end": df[C.TIME_COL].max(),
        "missing_cells": int(num.isna().sum().sum()),
        "duplicate_timestamps": int(df[C.TIME_COL].duplicated().sum()),
        "irregular_steps": int((dt != 1).sum()),
        "constant_columns": [c for c in num.columns if num[c].nunique() <= 1],
        "stage1_idle_rows": int(idle.sum()),
        "stage1_idle_start": df.loc[idle, C.TIME_COL].min() if idle.any() else None,
        "stage1_idle_end": df.loc[idle, C.TIME_COL].max() if idle.any() else None,
    }
