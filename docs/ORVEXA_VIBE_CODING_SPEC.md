# ORVEXA Vibe-Coding Specification

**Status:** Frozen implementation contract for a final-year AI and Data Science research prototype.

**Project title:** *ORVEXA: An Explainable and Calibrated Temporal AI Framework for Multi-Horizon Spacecraft Conjunction-Risk Prioritization*

**Primary rule:** The coding assistant must implement this specification exactly. It must not silently choose a different target, split, horizon rule, feature policy, model objective, or dashboard behavior. If a required field is absent or ambiguous, the program must stop with a clear validation error and request a configuration decision rather than silently guessing.

## 1. Project objective

ORVEXA takes historical ESA Conjunction Data Messages (CDMs), groups them into conjunction events, constructs leakage-free prefixes available at 2-, 3-, 5-, and optional 7-day warning horizons, predicts the final ESA event-level `risk` score, ranks events under fixed alert budgets, calibrates a secondary high-risk probability, explains predictions, and tests temporal and degraded-data robustness.

The primary AI model is a masked causal Temporal Convolutional Network (TCN). The mandatory comparison models are an ESA-derived `max_risk_estimate` baseline, Ridge/Logistic Regression, snapshot XGBoost, and temporal-summary XGBoost. The final dashboard is a research demonstration, not an operational collision-avoidance authority.

## 2. Non-negotiable scientific boundaries

The system must obey these rules:

1. The primary supervised dataset is ESA Collision Avoidance Challenge data.
2. The target is the final event-level ESA `risk` value, not an independently verified physical collision probability.
3. All input features must be available at or before the selected warning cutoff.
4. The final CDM and any post-cutoff CDMs must never appear in the input.
5. All CDM rows from one `event_id` must stay in one data partition.
6. The primary split must be ordered by the best available event-time/order key, not a random row split.
7. Imputers, scalers, encoders, thresholds, calibration maps, and model-selection decisions must be fit using training/validation data only.
8. Space-Track TLE data and CelesTrak data must remain separate from the ESA labelled ML table unless a rigorous object/time/frame mapping is proven.
9. An arbitrary TLE upload must not receive an ESA-trained risk score.
10. A negative TCN result is acceptable and must be reported honestly.

## 3. Official dataset sources

### 3.1 ESA Collision Avoidance Challenge: primary dataset

| Property | Contract |
|---|---|
| Official ESA page | https://kelvins.esa.int/collision-avoidance-challenge/data/ |
| Official archive | https://zenodo.org/records/4463683 |
| Zenodo DOI | `10.5281/zenodo.4463683` |
| File | `Collision Avoidance Challenge - Dataset.zip` |
| Primary training file | Extracted training CSV, normally named `train_data.csv`; discover by header and validate. |
| Expected training rows | 162,634; fail validation if materially different and require confirmation. |
| Expected columns | 103; fail validation if materially different and require confirmation. |
| Expected unique events | 13,154; report actual and compare. |
| Event key | `event_id` |
| Target column | `risk` |
| Horizon column | `time_to_tca`, days |
| Target rule | Risk from the final CDM of each event. |
| Target interpretation | ESA self-computed base-10 logarithmic risk value. Higher/less-negative values represent higher risk. |
| Training data measured earlier | 0 exact duplicate rows; 0 duplicate `(event_id, time_to_tca)` pairs; no missing `risk`; approximately 32.5% missing `c_rcs_estimate`. |

The official test set may be loaded only as a separate competition-style benchmark. It must not be used as the ordinary test set for all horizons because it is not a random sample and has special temporal restrictions. The main scientific results must use an event-level split created from the labelled training data.

### 3.2 Space-Track TLE History: supporting orbital source

| Property | Contract |
|---|---|
| Dataset page | https://huggingface.co/datasets/juliensimon/space-track-tle-history |
| Inspected mirror | https://huggingface.co/datasets/oxzoid/space-track-tle-history |
| Role | Separate TLE parsing, SGP4 propagation, and orbital visualisation only. |
| Main identifier | `norad_id` |
| Time | `epoch`, timezone-aware timestamp |
| Typical orbital fields | Inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion. |
| Other fields | Mean-motion derivative, BSTAR, international designator, `altitude_km`. |
| Integration | No row-wise join to ESA risk labels by default. |

### 3.3 CelesTrak: optional current catalogue source

| Property | Contract |
|---|---|
| Official page | https://celestrak.org/NORAD/elements/ |
| Example endpoint | `https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json` |
| Role | Current catalogue and TLE/OMM demonstration. |
| Retrieval rule | Save retrieval timestamp, endpoint, response hash, and record count. |
| ML scoring | Prohibited unless exact ESA-CDM feature compatibility is established. |

### 3.4 Excluded dataset

Do not use https://zenodo.org/records/4986292 in the core experiment. It is a debris-origin dataset and does not supply the ESA event-level conjunction-risk target.

## 4. Raw-data manifest contract

Create `data/manifests/dataset_manifest.yaml` with:

```yaml
esa:
  source_url: https://zenodo.org/records/4463683
  zenodo_doi: 10.5281/zenodo.4463683
  archive_name: Collision Avoidance Challenge - Dataset.zip
  retrieval_utc: "YYYY-MM-DDTHH:MM:SSZ"
  sha256: "fill-after-download"
  expected_rows: 162634
  expected_columns: 103
  expected_events: 13154
  license: "CC BY 4.0 as listed by the Zenodo record"
space_track:
  source_url: https://huggingface.co/datasets/juliensimon/space-track-tle-history
  retrieval_utc: "YYYY-MM-DDTHH:MM:SSZ"
  selected_files: []
celestrak:
  endpoint: https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json
  retrieval_utc: null
  response_sha256: null
```

Raw downloads belong in `data/raw/`, which must be Git-ignored. Never overwrite a raw file silently.

## 5. Exact ESA schema policy

The raw file must be validated against the actual header at runtime. The implementation must generate `reports/schema_report.json` containing every column, inferred dtype, missing count, missing fraction, minimum, maximum, and distinct count.

The safe feature policy is a **whitelist plus explicit controls**. Every raw column not present in the whitelist is excluded from model input by default. The assistant must not automatically use all 103 columns.

### 5.1 Mandatory target and control columns

| Column | Type | Unit/meaning | Policy |
|---|---|---|---|
| `event_id` | identifier | Anonymised conjunction event | Grouping only; never a feature. |
| `mission_id` | identifier/categorical | Mission/event identity field | Exclude from core model; may be audit-only because it can encode identity. |
| `risk` | float | Final/current ESA log-risk | Target only; never input. |
| `time_to_tca` | float | Days to TCA | Input if present at inference; retain as warning-context feature. |
| `c_object_type` | categorical | Object type | Encode consistently; retain if available at cutoff. |
| `max_risk_estimate` | float | ESA-derived maximum risk estimate | Controlled with/without ablation; never treat as unquestioned truth. |

### 5.2 Feature-group whitelist

The schema module must select available columns by exact name from the following group configuration. If a configured column is missing, record it as unavailable; do not substitute a similarly named column without an explicit alias map.

```yaml
feature_groups:
  encounter_geometry:
    - miss_distance
    - relative_speed
    - relative_position_r
    - relative_position_t
    - relative_position_n
    - relative_velocity_r
    - relative_velocity_t
    - relative_velocity_n
    - mahalanobis_distance
  conjunction_location:
    - geocentric_latitude
    - azimuth
    - elevation
  orbital_state:
    - t_a
    - t_e
    - t_i
    - t_argp
    - t_raan
    - t_mean_anomaly
    - t_perigee
    - t_apogee
    - c_a
    - c_e
    - c_i
    - c_argp
    - c_raan
    - c_mean_anomaly
    - c_perigee
    - c_apogee
  uncertainty_and_covariance:
    - t_sigma_r
    - t_sigma_t
    - t_sigma_n
    - c_sigma_r
    - c_sigma_t
    - c_sigma_n
    - miss_distance_sigma
    - covariance_determinant
    - covariance_correlation
  observation_quality:
    - t_obs_available
    - c_obs_available
    - t_obs_used
    - c_obs_used
    - t_od_span
    - c_od_span
    - t_rms
    - c_rms
    - t_residual
    - c_residual
  contextual:
    - time_to_tca
    - F10
    - F3M
    - AP
    - SSN
  controlled_physics:
    - max_risk_estimate
    - max_risk_scaling
```

Because public descriptions and versions may differ in exact capitalization or abbreviations, the first `data_audit` command must print the real header. The alias map must be manually confirmed against that header before training. Unknown columns are retained in the audit report but excluded.

### 5.3 Feature treatment

| Feature type | Treatment |
|---|---|
| Positive heavy-tailed scalar | Apply `log1p(x)` only when non-negative and the transform is recorded. |
| Signed heavy-tailed scalar | Apply signed log `sign(x)*log1p(abs(x))` only when configured. |
| Angle | Add sine/cosine representation; retain raw angle only if justified. |
| Categorical `c_object_type` | One-hot encode using train-fitted encoder with unknown handling. |
| ID | Exclude from model. |
| Missing numeric | Train-fitted median imputation plus a binary missingness indicator for static models; TCN receives imputed value plus mask channel. |
| `risk` | Never include in any input feature or temporal summary. |
| Final-CDM-only fields | Exclude from prefix input even if present in earlier rows when they are target-derived. |

## 6. Exact event-building contract

Implement `src/orvexa/event_builder.py` with this contract:

```python
def build_event_views(
    raw_df: pd.DataFrame,
    horizons_days: tuple[float, ...] = (2.0, 3.0, 5.0, 7.0),
) -> dict[str, pd.DataFrame]:
    """Return horizon-specific event snapshots and sequence prefixes."""
```

For each `event_id`:

1. Validate that `event_id`, `time_to_tca`, and `risk` exist.
2. Convert `time_to_tca` and `risk` to numeric; fail on unparseable non-null values.
3. Sort rows by the best available CDM creation/order field. The default must be explicitly configured in `configs/base.yaml`.
4. If no creation timestamp is available, sort within the event by descending `time_to_tca` so the earliest warning appears first and the latest approach appears last. Validate that the sequence is monotonic after sorting.
5. Identify the final row as the row with the smallest `time_to_tca` after validation. Verify that its `risk` is non-null.
6. Define the event target as the final row’s `risk`.
7. For horizon H, select rows with `time_to_tca >= H`.
8. The anchor row is the selected row with the smallest `time_to_tca` that still satisfies the condition.
9. The sequence prefix is all selected rows ordered oldest-to-newest and ending at the anchor.
10. Never use rows with `time_to_tca < H` in the input.
11. Store `event_id`, target, horizon, anchor time, sequence length, prefix duration, and source row indices.
12. Exclude the event from that horizon if no row satisfies the cutoff; report coverage.

Required invariants:

```text
min(prefix.time_to_tca) >= H
anchor.time_to_tca >= H
anchor.time_to_tca == min(prefix.time_to_tca)
target == risk(final_event_row)
max(prefix.source_row_index) is not used as an ordering shortcut
```

## 7. Exact split contract

The code must create one row per unique event before splitting. The event ordering strategy must be written to `reports/split_manifest.json`.

Preferred ordering:

1. An explicit CDM creation timestamp if it exists and is validated.
2. Otherwise the first appearance index of each event in the raw training file, with a warning that this is an ordered proxy rather than a true chronological timestamp.
3. Never use a random row split for the primary result.

Default boundaries are exact by event count:

```text
n_events = number of unique event_id
train_end = floor(0.70 * n_events)
validation_end = floor(0.85 * n_events)
train = ordered_events[:train_end]
validation = ordered_events[train_end:validation_end]
test = ordered_events[validation_end:]
```

The actual event IDs and boundary keys must be saved. If a validated event-date field is later discovered, use date boundaries instead and update the manifest. All horizon views must reuse the same event partition.

## 8. Target and classification contract

### 8.1 Regression

The regression target is the raw ESA log-risk value. Do not apply another logarithm. Use:

```yaml
target:
  name: risk
  transform: identity
  output_name: predicted_risk_log10
  loss: huber
  huber_delta: 1.0
```

The dashboard must display the output as **predicted final ESA log-risk**. It may additionally show `10 ** predicted_risk_log10` only with a clear label such as “risk-scale conversion for reference,” not as a validated physical probability.

### 8.2 Secondary classification

Classification is secondary and uses sensitivity analysis, not one claimed universal truth.

```yaml
classification:
  target_column: risk
  log10_thresholds: [-6.0, -5.0, -4.0]
  positive_rule: "risk >= threshold_log10"
  primary_report: "report all three thresholds"
  threshold_selection: "never selected using final test labels"
```

Use class-weighted Logistic Regression, XGBoost classification, and an optional TCN classification head. Report the positive count and prevalence for every threshold and split. Do not use accuracy as the headline metric.

## 9. Preprocessing contract

The exact sequence is:

```text
raw CSV
  ↓
validate schema and ranges
  ↓
construct event prefixes and assign event splits
  ↓
remove target, IDs, and prohibited fields
  ↓
fit numeric imputer on TRAIN only
  ↓
fit missingness indicator transformer on TRAIN only
  ↓
fit categorical encoder on TRAIN only
  ↓
fit configured transforms on TRAIN only
  ↓
fit scaler on TRAIN only for Ridge/MLP/TCN
  ↓
transform validation and test using frozen train objects
```

Tree models may use native missing values if configured, but the comparison must be documented. The default static pipeline is median imputation plus missingness indicators for consistent comparison. The TCN default is imputed values plus explicit masks.

Serialize preprocessing as `artifacts/preprocessors/{experiment_id}.joblib` and store the feature order in `artifacts/preprocessors/{experiment_id}_features.json`.

## 10. TCN input and architecture contract

### 10.1 Sequence format

```text
semantic order: oldest CDM → newest permissible CDM
storage before model: [batch, time, features]
pytorch-tcn input: [batch, channels/features, time]
padding: left padding so the newest CDM is aligned at the final position
mask: 1 for a real timestep, 0 for padding
maximum sequence length: 23 initially, based on measured maximum training event length
truncation: none in v1; if changed, retain the most recent 23 and report truncation
```

If the TCN package requires a different format, transpose explicitly and test the shape. The model must never infer padding as a real observation.

### 10.2 Frozen v1 architecture

```yaml
model: masked_causal_tcn
input_features: generated_from_feature_registry
channels: [64, 64, 128]
kernel_size: 3
dilations: [1, 2, 4]
activation: ReLU
normalization: weight_norm
causal: true
dropout: 0.15
readout: masked_mean_pooling
regression_head: Linear(128, 1)
classification_head: optional Linear(128, 1)
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 64
max_epochs: 100
early_stopping_patience: 12
random_seed: 42
regression_loss: Huber(delta=1.0)
classification_loss: weighted_BCE
```

Run a small validation grid only after v1 works:

```text
channels: [[32,32,64], [64,64,128]]
kernel_size: [3,5]
dropout: [0.10,0.15,0.30]
learning_rate: [1e-3,3e-4]
```

Select using validation MAE plus the primary ranking metric; never use final-test performance for tuning.

## 11. Mandatory experiment IDs

| ID | Model/configuration |
|---|---|
| A1 | Physics ranking using `max_risk_estimate` only |
| A2 | Ridge regression and Logistic Regression |
| A3 | Snapshot XGBoost without `max_risk_estimate` |
| A4 | Snapshot XGBoost with `max_risk_estimate` |
| A5 | Temporal-summary XGBoost without `max_risk_estimate` |
| A6 | Temporal-summary XGBoost with `max_risk_estimate` |
| A7 | TCN without `max_risk_estimate` |
| A8 | TCN with `max_risk_estimate` |
| A9 | Single snapshot versus temporal summaries versus TCN |
| A10 | Uncalibrated versus Platt versus isotonic calibration |
| A11 | Random group-aware versus ordered split, secondary only |
| A12 | Horizon sweep H = 2, 3, 5, 7 days |
| A13 | Clean versus missing/noisy inputs |
| A14 | Attribution stability by horizon and split |

Every run must save `experiment_id`, git commit, seed, config hash, data-manifest hash, split-manifest hash, feature list, sample counts, metrics, and artifact paths.

## 12. Evaluation contract

### Regression

Compute MAE, RMSE, median absolute error, Spearman correlation, and error by target-risk quantile.

### Classification

Compute PR-AUC, recall, precision, false negatives, and confusion matrix for each log threshold. ROC-AUC may be supplementary only.

### Alert-budget ranking

Rank by predicted final risk, with larger/less-negative values first. At K = 1%, 5%, and 10% of test events, compute:

```text
Recall@K = high-risk events selected in top K / all high-risk events
Precision@K = high-risk events selected in top K / number selected in top K
NDCG@K using risk-derived relevance
lift = Recall@K(model) / Recall@K(physics baseline)
missed_high_risk_events
absolute_review_count
```

Use the same test events and alert budget for all model comparisons.

### Calibration

Calibrate the secondary probability output using validation only. Compare uncalibrated, Platt, and isotonic. Use equal-frequency bins when positives are rare, report bin counts, Brier score, ECE, reliability curve, and calibration slope/intercept.

### Statistics

```yaml
bootstrap:
  unit: event
  iterations: 2000
  confidence_level: 0.95
  seed: 42
  interval: percentile
paired_comparison:
  unit: same_event
  report: mean_difference, median_difference, 95_percent_CI, practical_effect
```

Do not report row-bootstrap intervals because CDM rows within an event are dependent.

## 13. Robustness contract

Use the same perturbation seeds across models. Run:

| Test | Levels |
|---|---|
| Random feature masking | 10%, 20%, 40% |
| Remove observation-quality group | Yes/no |
| Remove uncertainty group | Yes/no |
| Proportional numerical noise | 1%, 5%, 10% |
| Sequence reduction | Latest 1, 3, and 5 valid CDMs |
| Internal CDM removal | Randomly remove valid internal timesteps |

Report clean metric, degraded metric, absolute degradation, relative degradation, and calibration degradation.

## 14. Explainability contract

For XGBoost use SHAP TreeExplainer. For TCN use Captum Integrated Gradients, feature occlusion, or grouped temporal perturbation. Attributions must be labelled as model behaviour, not causality.

Required outputs:

```text
artifacts/explanations/{experiment_id}/global_importance.csv
artifacts/explanations/{experiment_id}/local_event_{event_id}.json
artifacts/explanations/{experiment_id}/stability_by_horizon.csv
```

Report feature-level and group-level importance for encounter geometry, orbital state, uncertainty, observation quality, temporal summaries, and controlled physics features. Compare attribution stability across chronological windows.

## 15. Required command-line interface

Implement these commands:

```bash
python -m orvexa.cli audit-data --config configs/base.yaml
python -m orvexa.cli build-events --config configs/base.yaml
python -m orvexa.cli make-splits --config configs/base.yaml
python -m orvexa.cli train --experiment A1 --horizon 2 --config configs/base.yaml
python -m orvexa.cli train --experiment A8 --horizon 2 --config configs/base.yaml
python -m orvexa.cli evaluate --experiment A8 --horizon 2 --config configs/base.yaml
python -m orvexa.cli run-ablation --config configs/base.yaml
python -m orvexa.cli explain --experiment A8 --horizon 2 --config configs/base.yaml
streamlit run app/streamlit_app.py
```

Every command must fail with a useful message if the data manifest, schema, preprocessing artifact, or model artifact is missing.

## 16. Artifact contracts

### Prediction file

`artifacts/predictions/{experiment_id}_h{H}.parquet` must contain:

```text
event_id
split
horizon_days
anchor_time_to_tca
sequence_length
true_final_risk
predicted_final_risk_log10
high_risk_threshold_log10
true_high_risk
raw_probability_high_risk
calibrated_probability_high_risk
priority_rank
selected_top_1pct
selected_top_5pct
selected_top_10pct
model_version
```

### Metrics file

`artifacts/metrics/{experiment_id}_h{H}.json` must contain model metadata, event counts, coverage, all metrics, threshold, calibration method, seed, and confidence intervals.

### Model bundle

`artifacts/models/{experiment_id}_h{H}/` must contain:

```text
model.pt or model.json
preprocessor.joblib
feature_order.json
config.yaml
metrics.json
split_manifest.json
README.md
```

## 17. Dashboard contract

Use Streamlit first. The app must read precomputed prediction and metrics files; it must not retrain models during page interaction.

### Pages and controls

| Page | Controls | Required output |
|---|---|---|
| Overview | Horizon, experiment, split | Event/CDM counts, coverage, missingness, target distribution, limitations. |
| Ranked alerts | Horizon, model, alert budget 1/5/10%, threshold | Sortable table with event ID, predicted risk, calibrated probability, rank, TCA countdown, miss distance, target/chaser labels when available. |
| Event detail | Select event ID | CDM timeline, permissible prefix, target, prediction, probability, main features. |
| Explanation | Select event and model | SHAP/TCN feature/group attribution and non-causal disclaimer. |
| Reliability | Horizon, model | Reliability diagram, Brier, ECE, calibration slope/intercept, bin counts. |
| Horizon comparison | Model selector | Coverage and metrics for 2/3/5/7 days. |
| Robustness | Perturbation selector | Original versus masked/noisy prediction and metric degradation. |
| Orbital demo | TLE/OMM input or selected catalogue record | Source-labelled SGP4 propagation, epoch, units, warnings, altitude/position charts. |

### UI states

- If no artifact exists, show “Run the training/evaluation command first” and do not crash.
- If a selected horizon has no usable events, show coverage and disable ranking.
- If a CelesTrak record lacks ESA-compatible CDM features, show orbital information only and explicitly disable risk scoring.
- Every page must display: “Research estimate only. ORVEXA is not an operational collision-avoidance authority.”
- Add CSV/Parquet download buttons for the currently filtered ranking and metrics.

## 18. TLE/SGP4 module contract

Use the MIT-licensed `sgp4` Python package. Keep code in `src/orvexa/orbit/`. Validate NORAD ID, epoch, units, propagation interval, and SGP4 error code. Store source URL and retrieval timestamp. Display position/velocity units explicitly.

Do not calculate ESA ML risk from TLE-only input. The module may show propagated state, altitude, ground track, or orbital elements. It is a supporting physics demonstration and not part of the main supervised target.

## 19. Repository and dependency contract

Recommended dependencies:

```text
pandas
numpy
pyarrow
duckdb
scipy
scikit-learn
xgboost
torch
pytorch-tcn
shap
captum
matplotlib
seaborn
plotly
streamlit
sgp4
mlflow
pytest
ruff
```

Use [Kessler](https://github.com/kesslerlib/kessler) as a domain reference only unless GPL-3.0 obligations are intentionally accepted. Use [OrbVeil](https://github.com/ncdrone/orbveil) only as an optional Apache-2.0 CDM/screening adapter after pinning and testing a commit. Use [Python SGP4](https://github.com/brandon-rhodes/python-sgp4) and [PyTorch-TCN](https://github.com/paul-krug/pytorch-tcn) as dependencies or references. Record URLs, versions/commits, licenses, and modifications in `THIRD_PARTY_NOTICES.md`.

## 20. Unit-test contract

The following tests are mandatory:

| Test | Expected assertion |
|---|---|
| `test_target_is_final_risk` | Target equals risk from final event row. |
| `test_no_post_cutoff_rows` | Every prefix row has `time_to_tca >= H`. |
| `test_anchor_is_latest_qualifying_row` | Anchor has minimum qualifying `time_to_tca`. |
| `test_event_partition_disjoint` | Train/validation/test event sets are disjoint. |
| `test_no_target_in_features` | `risk` is absent from every feature list. |
| `test_no_id_features` | `event_id` and identifiers are absent from model features. |
| `test_train_only_preprocessing` | Validation/test transformation does not refit transformers. |
| `test_mask_shape` | TCN mask shape matches sequence batch and valid positions. |
| `test_padding_not_valid` | Padded positions are zero in the mask. |
| `test_ranking_direction` | Higher/less-negative predicted risk receives higher priority. |
| `test_bootstrap_is_event_level` | Resampling unit is event, never CDM row. |
| `test_arbitrary_tle_cannot_score` | TLE-only input does not invoke ESA risk prediction. |

## 21. Acceptance criteria

The implementation is complete only when:

1. A clean environment installs from the lock file.
2. Dataset audit reproduces expected rows, columns, events, missingness, and target summary.
3. Event construction is deterministic and passes all cutoff tests.
4. Split manifest lists exact event boundaries and no overlap.
5. A2, A3, A4, A5, A6, A7, and A8 train successfully for H=2.
6. All feasible horizons report usable-event coverage.
7. Prediction files and metrics files are generated with the required schemas.
8. Calibration is fit without final-test leakage.
9. Alert-budget ranking is evaluated on identical events for all models.
10. Robustness and explanation outputs are saved.
11. Dashboard loads artifacts and handles missing/empty states.
12. TLE/SGP4 module works separately with source and unit labels.
13. README contains reproducibility commands and limitations.
14. Every experiment records configuration, seed, commit, data hash, and artifact path.

## 22. Priority if time is limited

Implement in this order:

```text
1. Dataset audit and event builder
2. Leakage tests and event split
3. Physics baseline and XGBoost
4. Temporal summaries
5. TCN v1
6. Multi-horizon evaluation
7. Ranking metrics
8. Calibration
9. Robustness
10. Explainability
11. Dashboard
12. TLE/SGP4 polish
13. Optional Random Forest, MLP, LSTM/GRU
```

Do not sacrifice data correctness, leakage prevention, or baseline comparisons for additional algorithms or frontend polish.

## 23. Known limitations to include in the thesis

The ESA dataset is historical and anonymised. Its `risk` is a self-computed ESA log-risk value, not a directly observed collision outcome. The official competition test set is non-random and high-risk biased. Chronological ordering may require an ordered proxy if event timestamps are unavailable. TLE/SGP4 propagation has lower fidelity than operational high-precision ephemerides. The model is not certified for operational collision avoidance, maneuver decisions, or safety-critical use. Threshold-sensitive classification and alert-budget results must be interpreted with their event prevalence and confidence intervals.

## 24. Final novelty claim

Use this exact claim in the proposal:

> **ORVEXA contributes a reproducible, event-aware evaluation framework for temporal AI-based conjunction-risk prioritization. It tests whether a masked causal TCN adds value beyond ESA-derived and static ML baselines under multiple warning horizons, fixed alert budgets, calibration drift, explanation stability, and controlled missing/noisy observations. The work does not claim to invent collision-risk prediction or to replace operational space-safety systems.**

## References

[1]: https://kelvins.esa.int/collision-avoidance-challenge/data/ "ESA Kelvins Collision Avoidance Challenge data description"

[2]: https://zenodo.org/records/4463683 "Official ESA Collision Avoidance Challenge dataset, Zenodo record 10.5281/zenodo.4463683"

[3]: https://arxiv.org/html/2008.03069v2 "Uriot et al., Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition"

[4]: https://github.com/kesslerlib/kessler "Kessler collision-avoidance ML library"

[5]: https://github.com/ncdrone/orbveil "OrbVeil conjunction-screening engine"

[6]: https://github.com/brandon-rhodes/python-sgp4 "Python SGP4 library"

[7]: https://github.com/paul-krug/pytorch-tcn "PyTorch-TCN implementation"

[8]: https://huggingface.co/datasets/juliensimon/space-track-tle-history "Space-Track TLE History dataset"

[9]: https://celestrak.org/NORAD/elements/ "CelesTrak NORAD GP element sets"

[10]: https://scikit-learn.org/stable/modules/calibration.html "scikit-learn probability calibration documentation"

[11]: https://shap.readthedocs.io/ "SHAP documentation"

[12]: https://captum.ai/ "Captum documentation"

[13]: https://streamlit.io/ "Streamlit documentation"
