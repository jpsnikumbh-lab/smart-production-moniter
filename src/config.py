"""Central configuration: column layout of the dataset and shared constants.

The column ranges follow notes_on_dataset.txt (0-based column positions).
"""

TIME_COL = "time_stamp"
N_MEAS = 15            # number of output measurements per stage
EPS = 1e-3             # |value| <= EPS (mm) is treated as "no reading"
DEFAULT_TOL_PCT = 10.0 # default tolerance for the DERIVED quality label

# group name -> (first column, last column), inclusive
COLUMN_GROUPS = {
    "Ambient conditions": (1, 2),
    "Machine 1 - raw material": (3, 6),
    "Machine 1 - process": (7, 14),
    "Machine 2 - raw material": (15, 18),
    "Machine 2 - process": (19, 26),
    "Machine 3 - raw material": (27, 30),
    "Machine 3 - process": (31, 38),
    "Combiner (stage 1)": (39, 41),
    "Machine 4 - process": (72, 78),
    "Machine 5 - process": (79, 85),
}

STAGE1_FEATURE_RANGE = (1, 41)                     # everything upstream of stage-1 output
STAGE2_EXTRA_RANGES = [(72, 78), (79, 85)]         # machines 4 and 5


def actual_col(stage: int, i: int) -> str:
    return f"Stage{stage}.Output.Measurement{i}.U.Actual"


def setpoint_col(stage: int, i: int) -> str:
    return f"Stage{stage}.Output.Measurement{i}.U.Setpoint"
