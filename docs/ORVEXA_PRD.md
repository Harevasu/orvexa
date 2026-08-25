# ORVEXA Product Requirements Document

**Product:** ORVEXA

**Full title:** *An Explainable and Calibrated Temporal AI Framework for Multi-Horizon Spacecraft Conjunction-Risk Prioritization*

**Document status:** Product source of truth for implementation.

**Audience:** Final-year AI and Data Science project team, supervisor, coding assistant, evaluator, and dashboard users.

## 1. Product summary

ORVEXA is a research decision-support prototype that reads historical ESA Conjunction Data Messages, groups messages into conjunction events, predicts the final event-level ESA risk score from information available before a defined warning horizon, and ranks events for operator attention.

The product compares a physics-derived ESA estimate, static machine-learning models, temporal-summary models, and a masked causal Temporal Convolutional Network. It also provides probability calibration, model explanations, temporal-drift analysis, degraded-data robustness analysis, and a Streamlit dashboard.

ORVEXA is not an operational collision-avoidance system. It does not control spacecraft, recommend maneuvers, perform debris removal, or certify that a collision will or will not occur.

## 2. Product goals

| Goal | Success definition |
|---|---|
| Reliable event construction | Every model input is generated from valid CDMs before the selected cutoff, with no final-CDM leakage. |
| AI-centered research | TCN is trained and compared fairly against static and temporal-summary baselines. |
| Early warning | Performance and usable-event coverage are measured at 2, 3, 5, and optional 7 days before TCA. |
| Operator prioritization | Events are ranked under top-1%, top-5%, and top-10% alert budgets. |
| Trustworthy output | Probability calibration, explanations, temporal drift, and degraded-input behavior are quantified. |
| Reproducibility | Every run records configuration, seed, code version, data manifest, split manifest, metrics, and artifacts. |
| Understandable demonstration | A user can inspect the dataset, ranked alerts, a selected event, its explanation, reliability, and the separate TLE/SGP4 demonstration. |

## 3. Non-goals

ORVEXA will not perform computer vision, autonomous spacecraft maneuver planning, reinforcement learning, active debris removal planning, debris capture, collision avoidance certification, real-time operational safety decisions, or arbitrary TLE-to-risk prediction.

Space-Track TLE data and CelesTrak data are supporting orbital-data sources. They are not merged into the ESA-labelled training set by default.

## 4. Target users and personas

| Persona | Need | ORVEXA value |
|---|---|---|
| AI/DS student developer | A deterministic project that can be implemented, tested, and explained | Modular pipeline, configuration files, tests, experiment IDs, and reproducible artifacts. |
| Project supervisor | Evidence of a substantial AI and Data Science contribution | Clear research questions, baselines, TCN, ablations, metrics, limitations, and thesis-ready figures. |
| Research evaluator | A defensible comparison rather than an unsupported accuracy claim | Event-aware chronological validation, alert-budget metrics, calibration, robustness, and confidence intervals. |
| Space-safety analyst or domain reviewer | A ranked list with context and explanations | Event timeline, warning horizon, prediction, priority rank, feature contributions, and reliability indicators. |
| Demonstration viewer | A clear visual explanation of what the system does | Streamlit overview, ranked alerts, event detail, reliability, robustness, and orbital demo pages. |

## 5. Core product workflow

```text
Download and validate ESA data
        ↓
Group rows by event_id
        ↓
Order CDMs and identify final target risk
        ↓
Build leakage-free H=2/3/5/7 day prefixes
        ↓
Create event-aware chronological partitions
        ↓
Fit train-only preprocessing
        ↓
Train physics, statistical, XGBoost, temporal-summary, and TCN models
        ↓
Calibrate secondary probabilities
        ↓
Compute ranking, regression, calibration, robustness, and explanation outputs
        ↓
Save artifacts and metrics
        ↓
Display frozen artifacts in Streamlit
```

## 6. MVP scope

The MVP is complete when the following are implemented:

1. ESA archive download or local-file ingestion with manifest and checksum.
2. Schema audit reporting all columns, types, missingness, ranges, and event counts.
3. Event builder for 2-, 3-, and 5-day horizons; 7-day is attempted and marked unavailable if coverage is insufficient.
4. Event-aware chronological train/validation/test split.
5. Leakage tests proving that no prefix row is after its horizon cutoff.
6. Physics baseline using `max_risk_estimate`.
7. Ridge/Logistic Regression baselines.
8. Snapshot XGBoost with and without `max_risk_estimate`.
9. Temporal-summary XGBoost with and without `max_risk_estimate`.
10. Masked causal TCN with regression output and optional classification output.
11. Regression, PR-AUC, ranking, calibration, and coverage metrics.
12. Prediction and metric artifact files.
13. Streamlit pages for overview, ranked alerts, event detail, reliability, and horizon comparison.
14. README commands and automated unit tests.

## 7. Nice-to-have scope

The following may be implemented only after MVP results are stable:

- 7-day analysis if there is sufficient event coverage.
- Random Forest or MLP comparison.
- LSTM/GRU sensitivity model.
- Captum Integrated Gradients in addition to TCN occlusion.
- Rolling temporal-window evaluation.
- TLE/SGP4 orbital visualisation using Space-Track or CelesTrak.
- FastAPI service layer.
- React frontend.
- MLflow experiment UI.
- Docker Compose packaging.

Nice-to-have features must never delay leakage prevention, baseline comparisons, TCN training, or primary evaluation.

## 8. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The system shall load the ESA training CSV and validate required columns `event_id`, `time_to_tca`, and `risk`. | Must |
| FR-02 | The system shall group all rows by `event_id`. | Must |
| FR-03 | The system shall construct the final event target from the final CDM’s `risk`. | Must |
| FR-04 | The system shall construct prefixes using only rows with `time_to_tca >= H`. | Must |
| FR-05 | The system shall create separate event-level train, validation, and test partitions. | Must |
| FR-06 | The system shall fit preprocessing objects only on training data. | Must |
| FR-07 | The system shall train the physics, linear/logistic, XGBoost, temporal-summary, and TCN experiments. | Must |
| FR-08 | The system shall support model variants with and without `max_risk_estimate`. | Must |
| FR-09 | The system shall calculate regression, classification, ranking, calibration, and usable-coverage metrics. | Must |
| FR-10 | The system shall save predictions, metrics, configurations, and model artifacts. | Must |
| FR-11 | The system shall provide feature and event explanations with a non-causal disclaimer. | Must |
| FR-12 | The system shall test masking, noise, and reduced sequence conditions. | Should |
| FR-13 | The system shall provide a separate TLE/SGP4 orbital demonstration. | Should |
| FR-14 | The system shall display a visible research-only limitation on every dashboard page. | Must |
| FR-15 | The system shall prevent arbitrary TLE-only input from invoking ESA risk scoring. | Must |

## 9. User flows

### Flow A — Developer prepares data

The developer places the ESA archive or CSV in the configured raw-data location, runs `audit-data`, reviews the schema report, confirms the feature alias map, and runs `build-events`. The system produces horizon-specific event files and reports usable-event coverage. If required fields or expected counts are materially wrong, the process stops with an actionable error.

### Flow B — Developer trains a model

The developer selects an experiment ID and horizon, runs the `train` command, and receives a saved preprocessing artifact, model artifact, training history, configuration, and validation metrics. The system never reads the final test labels during tuning.

### Flow C — Developer evaluates results

The developer runs `evaluate`. The system writes prediction files, regression metrics, PR-AUC, alert-budget metrics, calibration metrics, event-level bootstrap intervals, and figures. The same event set and alert budget are used across comparable models.

### Flow D — Evaluator explores ranked alerts

The evaluator opens the dashboard, selects a horizon and model, chooses an alert budget, views the ranked event table, selects one event, and inspects its CDM timeline, prediction, probability, rank, feature values, explanation, and reliability label.

### Flow E — Evaluator compares models

The evaluator selects a horizon-comparison page and compares physics, snapshot XGBoost, temporal-summary XGBoost, and TCN on coverage, MAE, PR-AUC, Recall@K, Brier score, and ECE. The page states that results are research estimates.

### Flow F — Evaluator tests robustness

The evaluator chooses clean, masked, noisy, or reduced-sequence input. The dashboard displays prediction changes and metric degradation from precomputed artifacts. It does not retrain during interaction.

### Flow G — Evaluator views orbital demonstration

The evaluator selects a TLE/OMM record or a prepared sample. The system displays source, epoch, object ID, propagation warnings, position/velocity units, and orbital plots. It does not generate an ESA risk score from TLE-only input.

## 10. High-level acceptance criteria

The product is accepted only if:

- A clean checkout can reproduce the environment and run a data audit.
- The data audit reports 103 columns, approximately 162,634 training rows, and approximately 13,154 events for the validated ESA archive.
- No event occurs in more than one partition.
- No feature input contains `risk`.
- No input prefix contains a row below the selected `time_to_tca` cutoff.
- A2 through A8 produce reproducible artifacts for H=2.
- The TCN receives valid tensors and masks and trains without shape errors.
- Evaluation produces the required metric fields and confidence intervals.
- Calibration is fitted without final-test leakage.
- Dashboard filters work and empty states do not crash.
- The TLE module is visibly separate from ML risk scoring.
- The final report answers RQ1–RQ5 and documents negative or inconclusive results honestly.

## 11. Product success metrics

The project is not judged by a predetermined requirement that TCN must win. Success means that the study produces a trustworthy answer.

| Success area | Evidence |
|---|---|
| Scientific correctness | Leakage tests, split manifest, target audit, and reproducible pipeline. |
| AI depth | TCN versus XGBoost and temporal-summary ablations. |
| Operational relevance | Recall@1/5/10% and missed high-risk events. |
| Reliability | Brier, ECE, calibration curves, and drift analysis. |
| Transparency | Feature-group explanations and stability tables. |
| Engineering quality | Tests, configs, artifacts, and clean-install instructions. |
| Demonstration quality | Stable dashboard with clear limitations. |

## 12. Product limitations

The target is the final ESA log-risk score, not a directly observed collision outcome. The ESA official test set is not a random sample. The CDM data are anonymised, which may prevent rigorous TLE matching. SGP4/TLE propagation is a supporting demonstration and not an operational ephemeris solution. ORVEXA is not certified for safety-critical decisions.

## 13. Source links

- [ESA Kelvins Collision Avoidance Challenge data](https://kelvins.esa.int/collision-avoidance-challenge/data/)
- [ESA dataset Zenodo record](https://zenodo.org/records/4463683)
- [ESA competition paper](https://arxiv.org/html/2008.03069v2)
- [Space-Track TLE History](https://huggingface.co/datasets/juliensimon/space-track-tle-history)
- [CelesTrak GP data](https://celestrak.org/NORAD/elements/)
- [Kessler reference repository](https://github.com/kesslerlib/kessler)
- [Python SGP4](https://github.com/brandon-rhodes/python-sgp4)
- [PyTorch-TCN](https://github.com/paul-krug/pytorch-tcn)

## 14. Final product statement

> ORVEXA is a research prototype that uses temporal AI to estimate and prioritize the final ESA risk score of spacecraft-conjunction events from earlier CDM information. It compares physics-related, static ML, engineered temporal, and TCN models and evaluates prediction, ranking, calibration, explanations, temporal generalization, and degraded-data robustness.
