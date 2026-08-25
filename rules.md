# ORVEXA AI Context Rules

This file is the behavioral rulebook for any AI coding assistant working on ORVEXA. Read it before making changes.

## 1. Project identity

ORVEXA is an AI and Data Science research prototype for event-level spacecraft-conjunction risk prioritization. Its primary labelled data source is the ESA Collision Avoidance Challenge CDM dataset. Its primary AI model is a masked causal Temporal Convolutional Network. Its supporting models are an ESA-derived `max_risk_estimate` baseline, Ridge/Logistic Regression, snapshot XGBoost, and temporal-summary XGBoost.

The project predicts the final ESA `risk` score. Describe it as the **final ESA log-risk score**, not as a directly measured physical collision probability.

## 2. Mandatory technology stack

Use the following stack unless the user explicitly changes the architecture:

| Purpose | Technology |
|---|---|
| Language | Python 3.11+ |
| Data | Pandas, NumPy, PyArrow, DuckDB |
| Scientific computing | SciPy |
| Classical ML | scikit-learn |
| Tabular boosting | XGBoost |
| Deep learning | PyTorch |
| TCN | `pytorch-tcn` or a small local PyTorch implementation |
| Calibration | scikit-learn calibration utilities |
| Explainability | SHAP for trees; Captum or occlusion for TCN |
| Orbit support | Python `sgp4` package |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit first |
| Tracking | MLflow or structured JSON/CSV logs |
| Testing | Pytest |
| Quality | Ruff; mypy optional |
| Packaging | `pyproject.toml` and locked requirements |
| Version control | Git/GitHub |

Do not add React, FastAPI, Docker, a database, a Transformer, a GNN, or reinforcement learning unless a written requirement makes it necessary. Keep the first implementation local and reproducible.

## 3. Directory rules

Use this layout:

```text
orvexa/
├── README.md
├── LICENSE
├── CITATION.cff
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── requirements-lock.txt
├── .env.example
├── .gitignore
├── configs/
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── src/orvexa/
├── scripts/
├── tests/
├── notebooks/
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── explanations/
├── reports/
└── app/
```

Raw data must be Git-ignored. Do not place large archives or private files in Git without explicit approval.

## 4. Coding conventions

Use Python type hints for public functions. Use small functions with one responsibility. Prefer files below 400 lines; split modules when they grow larger. Use descriptive snake_case names, dataclasses or Pydantic-style validation for structured contracts, and docstrings for public APIs.

Do not hide errors with broad `except Exception`. Raise domain-specific errors with actionable messages. Validate inputs at module boundaries. Never silently coerce an unknown column, missing target, malformed timestamp, invalid horizon, or absent model artifact.

Use configuration files instead of hard-coding experiment decisions. Every training and evaluation command must record its configuration, seed, git commit, data hash, split hash, feature list, and output locations.

Do not modify a working scientific decision merely because a different implementation is easier. Explain any change in the commit or report.

## 5. Data and leakage rules

The ESA dataset sources are:

- https://kelvins.esa.int/collision-avoidance-challenge/data/
- https://zenodo.org/records/4463683

Expected validated training data are approximately 162,634 rows, 103 columns, and 13,154 unique events. The event key is `event_id`; the target is `risk`; the warning-time field is `time_to_tca` in days.

The final event target is the final CDM row’s `risk`. The input at horizon H contains only rows satisfying `time_to_tca >= H`. The final CDM must not be used as an input if it is below the cutoff. All rows from one event must remain in the same partition.

The primary split is event-aware and ordered: 70% earliest events for train, 15% next events for validation, 15% latest events for test. Save exact boundaries. If an explicit valid date field is available, use it; otherwise document the ordered proxy.

Never perform random row splitting as the primary result. Never fit an imputer, encoder, scaler, threshold, calibrator, or feature transform on validation or test data.

## 6. Feature rules

Use an explicit feature whitelist. At minimum, support these groups when the exact column names are confirmed by the audit report:

- Encounter geometry: miss distance, relative speed, relative position/velocity components, Mahalanobis distance.
- Encounter location: geocentric latitude, azimuth, elevation.
- Orbital state: target/chaser orbital variables, apogee, perigee, eccentricity, inclination, semi-major axis.
- Uncertainty/covariance: sigma variables, covariance correlations, covariance determinant.
- Observation quality: available/used observations, OD spans, residuals, RMS, observation windows.
- Context: `time_to_tca`, `F10`, `F3M`, `AP`, `SSN` when available.
- Controlled physics: `max_risk_estimate` and `max_risk_scaling` only in configured variants.

Always exclude `event_id`, identifiers, raw mission identity, and `risk` from model inputs. `max_risk_estimate` must have explicit with/without variants. Unknown columns are excluded by default and reported.

Treat missing numeric values with train-only imputation plus missingness indicators for static models. For TCN input, provide imputed values plus validity/missingness information. Do not replace missing values with zero without a configured scientific reason.

## 7. Target rules

Regression uses the identity-transformed ESA `risk` value, which is already logarithmic. Do not apply another log transform. TCN regression uses Huber loss with delta 1.0 in v1.

Secondary classification uses log-risk thresholds `[-6.0, -5.0, -4.0]` as sensitivity analysis. The label rule is `risk >= threshold_log10`. Report prevalence at every threshold. Never call one threshold universal unless the user explicitly provides an authoritative domain threshold.

## 8. TCN rules

The semantic sequence order is oldest CDM to newest permissible CDM. The raw tensor is `[batch, time, features]`; the PyTorch TCN tensor is `[batch, features, time]`. Use left padding, a validity mask with `1=real` and `0=padding`, and maximum sequence length 23 initially because that is the measured maximum event length in the validated training file.

Frozen v1 defaults:

```yaml
channels: [64, 64, 128]
kernel_size: 3
dilations: [1, 2, 4]
causal: true
dropout: 0.15
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 64
max_epochs: 100
early_stopping_patience: 12
seed: 42
loss: Huber(delta=1.0)
```

Use masked pooling or the final valid representation. Never allow padded timesteps to influence the output. Test tensor shape and mask behavior before training.

## 9. Experiment rules

Mandatory experiments are A1 through A14 as defined in `ORVEXA_VIBE_CODING_SPEC.md`. At minimum, implement A1, A2, A3, A4, A5, A6, A7, and A8 for H=2 before optional experiments.

The main comparison is physics → snapshot XGBoost → temporal-summary XGBoost → TCN. Do not collect algorithms without a research question. LSTM/GRU, MLP, Random Forest, Transformer, and GNN are optional or excluded; they are not required to make the project AI-focused.

## 10. Evaluation rules

The headline operational metric is Recall@1%, Recall@5%, and Recall@10% under equal event alert budgets. Also report MAE, RMSE, median absolute error, Spearman, PR-AUC, Precision@K, NDCG@K, missed high-risk events, Brier score, ECE, calibration slope/intercept, coverage, and 95% event-level bootstrap intervals.

Use 2,000 event-level bootstrap iterations with seed 42 and percentile intervals. Never bootstrap CDM rows as if they were independent observations.

## 11. Calibration and explanations

Fit Platt and isotonic calibrators on validation data only, select using validation Brier/ECE, and evaluate once on the final test. For tree models use SHAP TreeExplainer. For the TCN use Captum or feature/time occlusion. Attributions indicate model behavior, not causality. Report global importance, local event explanations, feature-group importance, and attribution stability by horizon/window.

## 12. Dashboard rules

Use Streamlit and precomputed artifacts. Never retrain on a page request. Required pages are Overview, Ranked Alerts, Event Detail, Explanation, Reliability, Horizon Comparison, Robustness, and Orbital Demonstration.

Every page must show: **Research estimate only. ORVEXA is not an operational collision-avoidance authority.**

If a selected artifact is missing, show an actionable empty state. If a horizon has no usable events, disable ranking. If a CelesTrak or TLE record is not ESA-CDM compatible, show orbit information only and disable ML scoring.

## 13. TLE rules

Space-Track TLE History is a separate supporting source:

https://huggingface.co/datasets/juliensimon/space-track-tle-history

CelesTrak is a separate optional current catalogue source:

https://celestrak.org/NORAD/elements/

Use Python SGP4 for parsing/propagation. Do not claim that TLE data are part of the ESA-labelled model unless a reproducible object-ID, time, frame, and encounter-feature mapping is proven.

## 14. Third-party code rules

Record each dependency in `THIRD_PARTY_NOTICES.md` with repository URL, version or commit, license, and purpose. Kessler is domain-relevant but GPL-3.0; do not copy its source without addressing the license. OrbVeil is optional and Apache-2.0; pin and test before use. Python SGP4 and PyTorch-TCN are suitable MIT-licensed dependencies.

## 15. Required workflow for every coding task

Before editing, read `ORVEXA_PRD.md`, `ORVEXA_TECHNICAL_DESIGN.md`, and `ORVEXA_TESTING.md`. Identify the affected requirement and acceptance criterion. Make the smallest change that satisfies the requirement. Add or update tests. Run relevant tests and a smoke test. Update documentation and artifact expectations. Never leave a change unexplained.

Before training, run the data audit and split validation. Before dashboard work, verify that frozen prediction artifacts exist. Before adding a new algorithm, state which research question it answers and why an existing model cannot answer it.

## 16. Forbidden shortcuts

Do not randomly split rows. Do not use the final risk as an input. Do not use post-cutoff rows. Do not fit transforms globally. Do not fill every missing value with zero. Do not join anonymised ESA CDMs to TLE rows without proof. Do not display arbitrary TLE data as an ESA risk prediction. Do not call predictions collision probabilities without qualification. Do not make the dashboard the novelty. Do not hide a negative result.
