"""Smoke tests: every page renders without an exception, and key logic behaves.
Run:  python -m pytest -q   (from the project folder)"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import anomaly, config as C, modeling, quality

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
CSV = ROOT / "data" / "continuous_factory_process.csv"

PAGES = ["1 · Production monitoring", "2 · Process parameters", "3 · Quality monitoring",
         "4 · Abnormality detection", "5 · Production analytics", "6 · Quality prediction (experimental)",
         "About & limitations"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV, parse_dates=[C.TIME_COL])


def test_zero_readings_are_not_valid(df):
    f = quality.measurement_frame(df, 1, 1)
    assert not f.loc[f["actual"] <= C.EPS, "valid"].any()
    assert f.loc[~f["valid"], "error"].isna().all()


def test_classification_is_derived_and_monotonic(df):
    f = quality.measurement_frame(df, 1, 0)
    tight = (quality.classify(f, 1) == quality.WITHIN).sum()
    loose = (quality.classify(f, 20) == quality.WITHIN).sum()
    assert loose >= tight


def test_anomaly_detects_injected_spike():
    s = pd.Series(np.random.default_rng(0).normal(10, 0.1, 1000))
    s.iloc[500] = 50
    assert anomaly.detect(s, "Global z-score", 4)[500]
    assert anomaly.detect(s, "Rolling z-score", 4)[500]
    assert anomaly.detect(s, "IQR fences", 3)[500]


def test_no_leakage_chronological_split(df):
    r = modeling.chrono_evaluate(df, 1, 3, "Ridge (linear)", 0.7)
    assert r["ok"]
    cut_time = df[C.TIME_COL].iloc[int(len(df) * 0.7)]
    assert (r["test_time"] >= cut_time).all()
