"""Quality-prediction experiments with HONEST validation.

* Samples are time-ordered, so a random split would leak information between
  neighbouring seconds. We train on the first part of the run and test on the
  later part (chronological split) and also use forward-chaining CV.
* Every model is compared with a naive baseline (predict the training mean).
* Only rows with a valid reading (see quality.py) are used.
* A model is labelled 'useful' only if it beats the baseline on unseen later
  data; otherwise the result is reported as NOT validated.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import feature_columns
from .quality import measurement_frame

MODELS = ["Ridge (linear)", "Random Forest"]
MIN_TRAIN, MIN_TEST = 200, 100
MIN_IMPROVEMENT_PCT = 5.0


def make_model(name: str, n_estimators: int = 100):
    if name == "Ridge (linear)":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=10,
                                 max_depth=14, n_jobs=-1, random_state=42)


def _verdict(r2: float, improvement_pct: float) -> str:
    if r2 > 0 and improvement_pct >= MIN_IMPROVEMENT_PCT:
        return "Beats baseline on unseen later data"
    return "NOT validated - does not beat the naive baseline"


def chrono_evaluate(df, stage, i, model_name, train_frac=0.7, n_estimators=100) -> dict:
    f = measurement_frame(df, stage, i)
    feats = feature_columns(df, stage)
    cut = int(len(df) * train_frac)
    pos = np.arange(len(df))
    tr = (f["valid"].to_numpy()) & (pos < cut)
    te = (f["valid"].to_numpy()) & (pos >= cut)
    if tr.sum() < MIN_TRAIN or te.sum() < MIN_TEST:
        return {"ok": False, "reason": f"Not enough valid readings (train {int(tr.sum())}, test {int(te.sum())}; "
                                       f"need {MIN_TRAIN}/{MIN_TEST})."}
    X, y = df[feats], f["actual"]
    model = make_model(model_name, n_estimators).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    base = np.full(te.sum(), y[tr].mean())
    mae, bmae = mean_absolute_error(y[te], pred), mean_absolute_error(y[te], base)
    r2 = r2_score(y[te], pred)
    imp = 100.0 * (bmae - mae) / bmae if bmae > 0 else np.nan
    if model_name == "Random Forest":
        imps = pd.Series(model.feature_importances_, index=feats)
    else:
        imps = pd.Series(np.abs(model[-1].coef_), index=feats)
    return {
        "ok": True, "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "r2": r2, "mae": mae, "rmse": float(np.sqrt(np.mean((y[te] - pred) ** 2))),
        "baseline_mae": bmae, "improvement_pct": imp, "verdict": _verdict(r2, imp),
        "test_time": f.loc[te, C.TIME_COL], "y_test": y[te], "pred": pd.Series(pred, index=y[te].index),
        "importance": imps.sort_values(ascending=False).head(10),
    }


def forward_cv(df, stage, i, model_name, n_splits=4, n_estimators=50) -> dict:
    """Expanding-window CV over valid rows, with a 60 s gap between train and test."""
    f = measurement_frame(df, stage, i)
    idx = np.flatnonzero(f["valid"].to_numpy())
    if len(idx) < (MIN_TRAIN + MIN_TEST) * 2:
        return {"ok": False}
    X, y = df[feature_columns(df, stage)].to_numpy(), f["actual"].to_numpy()
    r2s, imps = [], []
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=60).split(idx):
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            continue
        a, b = idx[tr], idx[te]
        m = make_model(model_name, n_estimators).fit(X[a], y[a])
        p = m.predict(X[b])
        bmae = mean_absolute_error(y[b], np.full(len(b), y[a].mean()))
        r2s.append(r2_score(y[b], p))
        imps.append(100.0 * (bmae - mean_absolute_error(y[b], p)) / bmae if bmae > 0 else np.nan)
    if not r2s:
        return {"ok": False}
    return {"ok": True, "folds": len(r2s), "r2_mean": float(np.mean(r2s)), "r2_worst": float(np.min(r2s)),
            "improvement_mean_pct": float(np.nanmean(imps)),
            "folds_beating_baseline": int(sum(1 for r, m in zip(r2s, imps) if r > 0 and m >= MIN_IMPROVEMENT_PCT))}


def all_targets_summary(df, stage, model_name, n_estimators=50, n_splits=4) -> pd.DataFrame:
    rows = []
    for i in range(C.N_MEAS):
        r = forward_cv(df, stage, i, model_name, n_splits, n_estimators)
        if r["ok"]:
            rows.append({"measurement": f"M{i}", "folds": r["folds"], "mean_R2": r["r2_mean"],
                         "worst_fold_R2": r["r2_worst"], "mean_MAE_improvement_pct": r["improvement_mean_pct"],
                         "folds_beating_baseline": f'{r["folds_beating_baseline"]}/{r["folds"]}',
                         "verdict": "Beats baseline in every fold" if r["folds_beating_baseline"] == r["folds"]
                         else "NOT validated"})
        else:
            rows.append({"measurement": f"M{i}", "folds": 0, "mean_R2": np.nan, "worst_fold_R2": np.nan,
                         "mean_MAE_improvement_pct": np.nan, "folds_beating_baseline": "-",
                         "verdict": "Too few valid readings"})
    return pd.DataFrame(rows)
