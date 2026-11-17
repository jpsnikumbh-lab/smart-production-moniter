"""Methodology and limitations."""
import streamlit as st


def render(df_full, df):
    st.header("About, methodology and limitations")
    st.markdown("""
**Project:** Smart Production Line Monitoring and Quality Prediction Using Industrial IoT
**Data:** Multi-Stage Continuous-Flow Manufacturing Process (Kaggle, `supergus`): 1 Hz process data with three
parallel machines, a combiner, and two series machines, with 15 output measurements after each stage.

### Measured vs derived
| Item | Kind | How it is obtained |
|---|---|---|
| Process variables, measurement *Actual*, *Setpoint* | Measured | Taken directly from the CSV |
| Error, relative error | Derived | actual − setpoint, only where the reading is valid |
| Within / outside tolerance | Derived | Rule set by the user (± % of setpoint) - **not** a recorded pass/fail |
| Anomaly flag | Derived | Statistical threshold on one variable at a time |
| Prediction | Modelled | Trained on early data, tested on later data, compared with a baseline |

### Data handling decisions
* An Actual value of exactly 0 is treated as **no reading** (a physical length in mm is not exactly 0).
* Rows where a setpoint is 0 are treated as **idle** and excluded from quality statistics.
* The dataset notes say setpoints are in the first row only; in the actual file they are present in every row and change
  during the run, so they are used row by row.

### Limitations
* The run is only about 3 h 55 min on a single day; behaviour in other conditions is unknown.
* Several measurements have many zero readings, so their statistics rest on few samples.
* Some measurements show a large, steady offset from their setpoint; this may reflect how setpoints map to sensors.
* Anomaly detection and tolerance classes cannot be scored for accuracy because there are no ground-truth labels.
* Correlation tables show association, not causation.
* Prediction results are reported on later unseen data; they should not be described as accurate unless the validation
  page shows the model beating the baseline.
""")
