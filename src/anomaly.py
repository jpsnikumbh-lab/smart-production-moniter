"""Statistical abnormality detection (no labels needed, no 'defect' claim).

A flagged point is statistically unusual for its own variable. It is a prompt to
inspect the process, not proof that a product is defective.
"""
import numpy as np
import pandas as pd

METHODS = ["Global z-score", "Rolling z-score", "IQR fences"]


def global_z(s: pd.Series, thr: float) -> pd.Series:
    sd = s.std()
    if not sd or np.isnan(sd):
        return pd.Series(False, index=s.index)
    return ((s - s.mean()) / sd).abs() > thr


def rolling_z(s: pd.Series, thr: float, window: int = 300) -> pd.Series:
    """Compare each point with the PREVIOUS `window` samples (excludes itself)."""
    mp = max(10, window // 2)
    mu = s.rolling(window, min_periods=mp).mean().shift(1)
    sd = s.rolling(window, min_periods=mp).std().shift(1).replace(0, np.nan)
    return ((s - mu) / sd).abs() > thr


def iqr_flags(s: pd.Series, k: float) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(False, index=s.index)
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def detect(s: pd.Series, method: str, thr: float, window: int = 300) -> pd.Series:
    if method == "Global z-score":
        return global_z(s, thr)
    if method == "Rolling z-score":
        return rolling_z(s, thr, window)
    return iqr_flags(s, thr)


def events_from_flags(time: pd.Series, values: pd.Series, flags: pd.Series, max_gap: int = 5) -> pd.DataFrame:
    """Merge flagged samples closer than `max_gap` rows into one event."""
    pos = np.flatnonzero(flags.to_numpy())
    cols = ["start", "end", "duration_s", "flagged_points", "peak_value", "median_value"]
    if len(pos) == 0:
        return pd.DataFrame(columns=cols)
    med = values.median()
    t = time.reset_index(drop=True)
    v = values.reset_index(drop=True)
    rows = []
    for g in np.split(pos, np.where(np.diff(pos) > max_gap)[0] + 1):
        seg = v.iloc[g]
        peak = seg.iloc[int((seg - med).abs().argmax())]
        rows.append({
            "start": t.iloc[g[0]], "end": t.iloc[g[-1]],
            "duration_s": (t.iloc[g[-1]] - t.iloc[g[0]]).total_seconds() + 1,
            "flagged_points": len(g), "peak_value": peak, "median_value": med,
        })
    return pd.DataFrame(rows, columns=cols)
