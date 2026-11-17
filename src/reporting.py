"""Standalone HTML report (no external assets) for the selected window."""
import html
from datetime import datetime

import pandas as pd

from . import anomaly, config as C, quality

CSS = """body{font-family:Arial,Helvetica,sans-serif;max-width:1000px;margin:30px auto;color:#222;line-height:1.45}
h1{color:#1f3a5f}h2{color:#1f3a5f;border-bottom:2px solid #dde6f2;padding-bottom:4px;margin-top:32px}
table{border-collapse:collapse;font-size:12px;width:100%}th,td{border:1px solid #ccd;padding:4px 6px;text-align:right}
th{background:#eef3fb}td:first-child,th:first-child{text-align:left}.note{background:#fff8e1;border-left:4px solid #f0ad4e;padding:8px 12px}"""


def _tbl(df: pd.DataFrame) -> str:
    return df.round(2).to_html(index=False, border=0, na_rep="-")


def build_report_html(df, health, tol, s1, s2, comparison) -> str:
    t0, t1 = df[C.TIME_COL].min(), df[C.TIME_COL].max()
    # abnormality overview for the 15 stage-1 measurements (rolling z-score, 4 sigma)
    rows = []
    for i in range(C.N_MEAS):
        s = quality.measurement_frame(df, 1, i)["actual_valid"]
        fl = anomaly.rolling_z(s, 4.0).fillna(False)
        rows.append({"measurement": f"M{i}", "valid_points": int(s.notna().sum()), "flagged_rolling_z4": int(fl.sum())})
    an = pd.DataFrame(rows)
    low1 = ", ".join(s1.loc[s1["coverage_pct"] < 50, "measurement"]) or "none"
    biased = ", ".join(s1.loc[(s1["bias_mm"].abs() > 0.2 * s1["setpoint_mm"]), "measurement"]) or "none"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Production quality report</title><style>{CSS}</style></head><body>
<h1>Smart Production Line Monitoring - Report</h1>
<p>Generated {datetime.now():%Y-%m-%d %H:%M}. Window: {t0} to {t1} ({len(df):,} records at 1 Hz).</p>
<div class="note"><b>Measured vs derived.</b> Actual and setpoint values are measured data. "Within tolerance" is a <i>derived</i>
label (|actual - setpoint| &le; {tol:g}% of setpoint); the dataset has no pass/fail column. Anomaly flags are statistical,
not confirmed defects. No prediction accuracy is claimed in this report.</div>
<h2>1. Data overview</h2>
<p>Missing cells: {health['missing_cells']}; duplicate timestamps: {health['duplicate_timestamps']};
non-1-second steps: {health['irregular_steps']}; stage-1 idle rows (all setpoints 0): {health['stage1_idle_rows']}.</p>
<h2>2. Stage 1 (primary) quality summary</h2>{_tbl(s1)}
<p>Measurements with under 50% usable readings: {html.escape(low1)}. Measurements with a persistent offset larger than 20% of the setpoint: {html.escape(biased)}.</p>
<h2>3. Stage 2 (secondary) quality summary</h2>{_tbl(s2)}
<h2>4. Stage 1 vs Stage 2</h2>{_tbl(comparison)}
<h2>5. Abnormality overview (stage 1, rolling z-score, 4 sigma, 300 s window)</h2>{_tbl(an)}
</body></html>"""
