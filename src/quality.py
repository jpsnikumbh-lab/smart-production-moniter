"""Quality metrics.

MEASURED data  : the *.U.Actual and *.U.Setpoint columns, exactly as recorded.
DERIVED labels : everything produced here (error, status, tolerance class). The
                 dataset has NO pass/fail column, so the tolerance class is an
                 analyst-defined rule, not a ground-truth quality label.

Assumption: an Actual value of exactly 0 (|x| <= EPS) is a missing/failed reading,
because a physical dimension in mm is not exactly zero. A Setpoint of 0 means the
line was idle/not targeting that feature, so those rows are excluded as well.
"""
import numpy as np
import pandas as pd

from . import config as C

WITHIN = "Within tolerance (derived)"
OUTSIDE = "Outside tolerance (derived)"
NO_READING = "No reading (actual = 0)"
IDLE = "Idle (setpoint = 0)"


def measurement_frame(df: pd.DataFrame, stage: int, i: int) -> pd.DataFrame:
    a = df[C.actual_col(stage, i)].to_numpy(dtype=float)
    sp = df[C.setpoint_col(stage, i)].to_numpy(dtype=float)
    idle = sp <= C.EPS
    noread = (~idle) & (a <= C.EPS)
    valid = ~(idle | noread)
    status = np.where(idle, "idle", np.where(noread, "noreading", "valid"))
    error = np.where(valid, a - sp, np.nan)
    rel = np.where(valid, (a - sp) / np.where(sp > C.EPS, sp, np.nan) * 100.0, np.nan)
    return pd.DataFrame({
        C.TIME_COL: df[C.TIME_COL].to_numpy(),
        "actual": a, "setpoint": sp,
        "actual_valid": np.where(valid, a, np.nan),
        "status": status, "valid": valid,
        "error": error, "rel_error_pct": rel,
    }, index=df.index)


def classify(frame: pd.DataFrame, tol_pct: float) -> pd.Series:
    """DERIVED label for every row (see module docstring)."""
    lab = pd.Series(NO_READING, index=frame.index, dtype=object)
    lab[frame["status"] == "idle"] = IDLE
    within = frame["valid"] & (frame["rel_error_pct"].abs() <= tol_pct)
    lab[within] = WITHIN
    lab[frame["valid"] & ~within] = OUTSIDE
    return lab


def stage_summary(df: pd.DataFrame, stage: int, tol_pct: float) -> pd.DataFrame:
    rows = []
    for i in range(C.N_MEAS):
        f = measurement_frame(df, stage, i)
        active = (f["status"] != "idle").sum()
        v = f[f["valid"]]
        n = len(v)
        sp = v["setpoint"].median() if n else np.nan
        rows.append({
            "measurement": f"M{i}",
            "setpoint_mm": sp,
            "coverage_pct": 100.0 * n / active if active else np.nan,
            "mean_actual_mm": v["actual"].mean() if n else np.nan,
            "bias_mm": v["error"].mean() if n else np.nan,
            "std_error_mm": v["error"].std() if n > 1 else np.nan,
            "mae_mm": v["error"].abs().mean() if n else np.nan,
            "rmse_mm": float(np.sqrt((v["error"] ** 2).mean())) if n else np.nan,
            "mae_pct_of_setpoint": v["rel_error_pct"].abs().mean() if n else np.nan,
            "within_tol_pct": 100.0 * (v["rel_error_pct"].abs() <= tol_pct).mean() if n else np.nan,
            "valid_rows": n,
        })
    return pd.DataFrame(rows)
