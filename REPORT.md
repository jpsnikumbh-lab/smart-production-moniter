# Smart Production Line Monitoring and Quality Prediction Using Industrial IoT

**Submitted by:** [Jaypalsing Nikumbh (231040035) ]  **Course:** [MDM- 5 Industrial IOT ]

## Abstract
This project builds a Streamlit dashboard for a multi-stage continuous-flow manufacturing line using the public Kaggle
dataset *Multi-Stage Continuous-Flow Manufacturing Process* (14,088 records at 1 Hz, 116 variables). The dashboard has
five monitoring/analytics modules and an experimental prediction module. The dataset contains measurements and
setpoints but no pass/fail label, so quality classes shown in the dashboard are derived with an explicit tolerance rule
and are labelled as such. Prediction models were tested on later, unseen data and compared with a naive baseline. In this
dataset **no model beat the baseline**, so no prediction accuracy is claimed. The most useful results are descriptive:
which measurements have usable readings, which sit persistently away from their setpoint, and how stage 2 compares with
stage 1.

## 1. Objectives
1. Monitor production records and the processing stages.
2. Visualise actual manufacturing measurements.
3. Analyse product quality and measurement variation.
4. Identify unusual measurements with statistical thresholds.
5. Compare stages and generate a report.
6. Test whether quality measurements can be predicted, and report the result honestly.

## 2. Dataset
* Source: Kaggle, *Multi-Stage Continuous-Flow Manufacturing Process* (supergus).
* 14,088 rows × 116 columns; 6 March 2019, 10:52:33 to 14:47:20 (about 3 h 55 min); no missing cells.
* Process: Machines 1-3 in parallel → combiner → 15 stage-1 measurements → Machines 4 and 5 in series → 15 stage-2
  measurements. Each measurement has an actual value and a setpoint (mm).
* Naming: `C` = controlled variable, `U` = uncontrolled variable.

Data observations (computed by the app):
* 14 duplicate timestamps and 28 non-1-second steps.
* All 15 stage-1 setpoints are 0 in 56 rows (idle/changeover); stage-2 setpoints are constant.
* Many actual values are exactly 0. Stage-1 M5, M7 and M11 have usable readings in under 50% of active rows; stage-2 M4
  has 9%.
* The supplied notes say setpoints appear only in the first row; the file actually holds a setpoint in every row.

## 3. System design
Python, pandas, NumPy, Plotly, scikit-learn and Streamlit. `src/` holds the logic (loading, quality metrics, anomaly
detection, modelling, report) and `views/` holds one page per module, so each module can be read and tested on its own.
`tests/test_smoke.py` renders every page and checks key behaviours (zero readings are excluded, a spike is detected, the
train/test split is chronological).

## 4. Methods
**Measured vs derived.** Actual and setpoint values are measured. Error = actual − setpoint, computed only for valid
readings. The tolerance class (within / outside ±x% of setpoint, default 10%) is a derived, user-adjustable rule and not
a recorded outcome.

**Valid readings.** Actual = 0 is treated as no reading and setpoint = 0 as idle. This is an assumption; it is shown on
the dashboard.

**Abnormality detection.** Three unsupervised methods on a single variable: global z-score, rolling z-score (each point
compared with the previous 300 s) and IQR fences. Consecutive flags are merged into events. Because there are no labelled
defects, detection accuracy cannot be measured and is not claimed; flags indicate where to inspect.

**Prediction.** Inputs are the upstream process variables (columns 1-41, plus Machines 4-5 for stage 2). Models: Ridge
regression and Random Forest. Validation: (a) chronological 70/30 split and (b) four-fold expanding-window
cross-validation with a 60 s gap. Baseline: the training-period mean. A model is called useful only if R² > 0 and MAE is at least
5% below the baseline on unseen later data. Random splits were not used because neighbouring seconds are highly similar and
would leak information.

## 5. Results
All figures below use the full run and a 10% tolerance; the dashboard recomputes them for any window.

**Quality (stage 1, primary).**
* Measurements with a persistent offset larger than 20% of the setpoint: M1, M6, M12. Under the 10% rule only 0%, 4%
  and 9% of their valid readings are within tolerance, which points to a systematic gap between setpoint and sensor value
  rather than random noise.
* M0, M3, M4, M8 and M9 have 99-100% of their valid readings within ±10%. M5, M7 and M11 have 100% or close to it within
  tolerance but rest on few readings (under 50% coverage), so they should be treated with caution.

**Stage 1 vs stage 2.** Stage 2 has a larger error spread (std in mm) than stage 1 for 8 of 15 measurements, so the "stage 2 is
noisier" statement in the dataset notes is only partly visible measurement by measurement. The correlation between the stage-1
and stage-2 actual values is weak for every measurement (|r| ≤ 0.22).

**Drivers.** Linear correlations between upstream variables and stage-1 measurements are small (for M3 the largest is
|r| = 0.17, Machine 3 zone-1 temperature). Correlation is association, not cause.

**Prediction.** Ridge regression was validated with four-fold forward-chaining on all 15 stage-1 and all 15 stage-2 targets, and
Random Forest (30 trees, for speed) on the 15 stage-1 targets. In no case did a model beat the baseline in every fold: for Ridge the
mean cross-validated R² was below 0 for all 30 targets. Random Forest showed a lower average MAE than the baseline for some targets
(for example M4, M11 and M12) but with R² below 0, so these are reported as **not validated**. A single chronological
70/30 split gave the same picture for the targets tested. No prediction accuracy is claimed.

## 6. Limitations
* One run of about four hours on a single day.
* Heavy zero readings limit several measurements; treating zero as "no reading" is an assumption.
* The tolerance class and anomaly flags have no ground-truth labels to be scored against.
* Prediction used simple models without feature engineering or lagged inputs; other approaches may do better, and this
  work does not show that quality is unpredictable, only that these models did not generalise to later data in this run.
* Correlation tables do not establish causes.

## 7. Conclusion and future work
The dashboard delivers the five requested modules and keeps measured data, derived labels and modelled output separate.
The honest finding is that, in this single run, descriptive monitoring is reliable while prediction is not validated.
Future work: lagged and rolling features, physical delay between stages, per-regime models, more than one production
run, labelled defects to evaluate anomaly detection, and real-time streaming input.

## 8. How to run
See `README.md`.

## 9. App link
link- "https://smart-appuction-moniter-jqwezeyq4gxmsmaxgabyhm.streamlit.app/"

## References
* Kaggle dataset: *Multi-Stage Continuous-Flow Manufacturing Process*, https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process
* Streamlit, Plotly, scikit-learn, pandas documentation.
