# Smart Production Line Monitoring and Quality Prediction Using Industrial IoT

A Streamlit application that monitors a multi-stage continuous-flow manufacturing line using the Kaggle dataset
*Multi-Stage Continuous-Flow Manufacturing Process* (14,088 one-second records, 116 variables, 6 March 2019).

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Python 3.10+ is recommended. The dataset is already in `data/`. A different CSV with the same columns can be loaded
from the sidebar.

Run the tests (every page is rendered automatically and the core logic is checked):
```bash
python -m pytest -q
```

## Modules

| # | Page | What it does |
|---|------|--------------|
| 1 | Production monitoring | Record counts, process-flow diagram, sensor availability over time, data-health checks, record explorer |
| 2 | Process parameters | Trends, rolling means, statistics, distributions and correlations of the machine/ambient variables |
| 3 | Quality monitoring | Actual vs setpoint, error distribution, user-set tolerance band, summary of all 15 measurements per stage |
| 4 | Abnormality detection | Global z-score, rolling z-score and IQR fences; merged event list; scan across all variables |
| 5 | Production analytics | Stage 1 vs Stage 2 comparison, correlation drivers, downloadable HTML report |
| 6 | Quality prediction (experimental) | Ridge / Random Forest tested on later unseen data against a naive baseline |
| - | About & limitations | Methodology and what the results do and do not show |

## Measured vs derived (important)

The dataset has **no pass/fail label**. The app keeps the two apart:

* **Measured:** process variables, measurement `Actual` and `Setpoint`, exactly as in the CSV.
* **Derived:** error (actual − setpoint), "within/outside tolerance" (a rule with a tolerance % chosen by the user),
  anomaly flags (statistical thresholds). These are labelled "derived" in the interface.
* **Modelled:** predictions. They are validated chronologically and compared with a baseline; the app states plainly when a
  model does **not** beat it.

## Data-handling decisions

* Actual = 0 is treated as *no reading* (a length in mm is not exactly zero); setpoint = 0 is treated as *idle*.
* The dataset notes say setpoints are only in the first row; in the file they are present in every row and change during
  the run (all 15 stage-1 setpoints go to 0 in 56 rows), so they are used row by row.
* Charts are down-sampled to about 3,000-4,000 points for speed; statistics use all rows.

## Project layout

```
app.py              entry point, sidebar, time-window filter
src/config.py       column layout and constants
src/data.py         loading, groups, data-health checks
src/quality.py      measurement frames, derived tolerance label, summaries
src/anomaly.py      detection methods and event merging
src/modeling.py     chronological split, forward-chaining CV, baseline
src/reporting.py    HTML report
views/              one file per page
tests/              smoke tests
data/               dataset and notes
REPORT.md           written project report
```
