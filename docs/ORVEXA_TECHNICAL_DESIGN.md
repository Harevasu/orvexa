# ORVEXA Technical Design Document

## 1. Architecture overview

ORVEXA is a batch research pipeline with a read-only dashboard over precomputed artifacts.

```text
ESA archive/CSV
    ↓
Data audit and schema registry
    ↓
Event builder and horizon prefixes
    ↓
Event-aware chronological split
    ↓
Train-only preprocessing
    ↓
Physics / linear / XGBoost / temporal-summary / TCN models
    ↓
Calibration, ranking, explanations, robustness
    ↓
Parquet/JSON/PNG artifacts
    ↓
Streamlit dashboard

Space-Track/CelesTrak
    ↓
Separate TLE/OMM adapter
    ↓
SGP4 propagation and orbital plots only
```

The dashboard must never retrain models. All expensive operations happen through CLI scripts.

## 2. Components

| Component | Module | Responsibility |
|---|---|---|
| Configuration | `src/orvexa/config.py` | Load and validate YAML configuration. |
| Data I/O | `src/orvexa/data_io.py` | Load CSV/Parquet and write manifests. |
| Schema | `src/orvexa/schema.py` | Required-column validation, feature registry, audit report. |
| Quality | `src/orvexa/quality.py` | Missingness, duplicates, ranges, event counts. |
| Event builder | `src/orvexa/event_builder.py` | Final target and horizon prefixes. |
| Splitting | `src/orvexa/splitting.py` | Event-aware chronological partitions. |
| Preprocessing | `src/orvexa/preprocessing.py` | Train-fitted imputation, encoding, transforms, scaling. |
| Snapshot features | `src/orvexa/features_snapshot.py` | Latest qualifying CDM representation. |
| Temporal features | `src/orvexa/features_temporal.py` | Deltas, slopes, ranges, coverage and missingness summaries. |
| Sequence dataset | `src/orvexa/datasets.py` | Padded tensors, masks, PyTorch DataLoaders. |
| Linear models | `src/orvexa/models_linear.py` | Ridge/Elastic Net and Logistic Regression. |
| XGBoost | `src/orvexa/models_xgb.py` | Snapshot and temporal-summary models. |
| TCN | `src/orvexa/models_tcn.py` | Masked causal TCN, regression/classification heads. |
| Calibration | `src/orvexa/calibration.py` | Platt/isotonic calibration. |
| Ranking | `src/orvexa/ranking_metrics.py` | Recall@K, Precision@K, NDCG, lift. |
| Robustness | `src/orvexa/robustness.py` | Masking, noise, reduced sequences. |
| Explainability | `src/orvexa/explainability.py` | SHAP, Captum/occlusion, stability. |
| Reporting | `src/orvexa/reporting.py` | Metrics, bootstrap intervals, figures. |
| Orbit | `src/orvexa/orbit/` | TLE/OMM validation and SGP4 propagation. |
| CLI | `src/orvexa/cli.py` | Stable commands and error messages. |
| App | `app/streamlit_app.py` | Read-only visualization of artifacts. |

## 3. Configuration schema

Create `configs/base.yaml`:

```yaml
project:
  name: ORVEXA
  seed: 42
  timezone: UTC
paths:
  raw_esa: data/raw/esa/train_data.csv
  processed: data/processed
  artifacts: artifacts
horizons_days: [2.0, 3.0, 5.0, 7.0]
data:
  required_columns: [event_id, time_to_tca, risk]
  event_key: event_id
  cutoff_column: time_to_tca
  target_column: risk
  max_sequence_length: 23
split:
  train_fraction: 0.70
  validation_fraction: 0.15
  test_fraction: 0.15
  ordering: first_valid_event_timestamp_else_first_appearance
classification:
  thresholds_log10: [-6.0, -5.0, -4.0]
  positive_rule: risk >= threshold_log10
metrics:
  alert_budgets: [0.01, 0.05, 0.10]
  bootstrap_iterations: 2000
  confidence_level: 0.95
  bootstrap_seed: 42
training:
  max_epochs: 100
  early_stopping_patience: 12
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
```

Every experiment has its own file under `configs/experiments/` and must override only what differs.

## 4. Data contracts

### Event record

```python
@dataclass
class EventRecord:
    event_id: str
    horizon_days: float
    target_risk: float
    anchor_time_to_tca: float
    sequence_length: int
    prefix_duration_days: float
    row_indices: list[int]
```

### Prepared event table

Each horizon-specific table must include:

```text
event_id
horizon_days
anchor_time_to_tca
target_risk
sequence_length
prefix_duration_days
prefix_row_ids
split
```

The sequence values may be stored separately in a long-form Parquet table keyed by `event_id` and `horizon_days`.

### Prediction schema

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

### Metrics schema

```json
{
  "experiment_id": "A8",
  "horizon_days": 2.0,
  "split": "test",
  "n_events": 0,
  "usable_coverage": 0.0,
  "mae": null,
  "rmse": null,
  "spearman": null,
  "pr_auc": null,
  "brier": null,
  "ece": null,
  "recall_at_1pct": null,
  "recall_at_5pct": null,
  "recall_at_10pct": null,
  "bootstrap": {"iterations": 2000, "confidence_level": 0.95, "seed": 42}
}
```

## 5. Event-building algorithm

`build_event_views()` must validate required columns, convert numeric values, group by `event_id`, sort by explicit creation time if available, otherwise sort descending by `time_to_tca`, identify the minimum-time final row, and build each prefix.

The final target is read only from the final row. The target is joined to a prefix after the prefix has been constructed. For each H, select rows where `time_to_tca >= H`; the anchor is the qualifying row with the smallest `time_to_tca`; the sequence is ordered oldest-to-newest and ends at the anchor.

The builder must emit an invariant report. Any violation stops the run.

## 6. Preprocessing architecture

Use separate fitted objects per model family when necessary:

```text
FeatureRegistry
  → ColumnTransformer
      numeric: median imputer + missing indicators + configured transforms + optional scaler
      categorical: most-frequent imputer + one-hot encoder(handle_unknown=ignore)
```

Fit only on training events. For temporal sequences, fit feature statistics on flattened training-prefix values only, then apply them to validation/test sequences. Store feature order and mask order.

## 7. Model interfaces

Every model must implement:

```python
class RiskModel(Protocol):
    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> "RiskModel": ...
    def predict_risk(self, X) -> np.ndarray: ...
    def predict_probability(self, X, threshold_log10: float) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "RiskModel": ...
```

The physics model implements `predict_risk` from `max_risk_estimate` with the documented direction and does not train.

## 8. TCN data interface

Store sequence tensors as `(N, T, F)` before model entry and transpose to `(N, F, T)` for the PyTorch TCN package. Store masks as `(N, T)` with `True` for valid timesteps. Left-pad to T=23. For an event shorter than 23, the newest valid CDM must be at index 22.

Use masked mean pooling:

```python
valid = mask.unsqueeze(1).float()
pooled = (hidden * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
```

## 9. API/data-service contract

The MVP needs no network API. The Streamlit app reads local artifacts. If FastAPI is added later, implement:

| Method | Endpoint | Request | Response |
|---|---|---|---|
| GET | `/health` | None | `{"status":"ok","version":"..."}` |
| GET | `/experiments` | None | Available experiments/horizons. |
| GET | `/metrics` | `experiment_id`, `horizon`, `split` | Metrics JSON. |
| GET | `/predictions` | `experiment_id`, `horizon`, `budget` | Filtered prediction rows. |
| GET | `/events/{event_id}` | `experiment_id`, `horizon` | Event details and CDM timeline. |
| GET | `/explanations/{event_id}` | `experiment_id`, `horizon` | Stored attribution payload. |
| GET | `/robustness` | `experiment_id`, `horizon` | Degradation table. |
| POST | `/orbit/propagate` | TLE/OMM and times | Orbit state and warnings; never risk score. |

All endpoints must return structured error JSON and must not train models.

## 10. Artifact naming

Use deterministic names:

```text
artifacts/models/A8_h2.0/model.pt
artifacts/models/A8_h2.0/preprocessor.joblib
artifacts/predictions/A8_h2.0_test.parquet
artifacts/metrics/A8_h2.0_test.json
artifacts/figures/A8_h2.0_reliability.png
artifacts/explanations/A8_h2.0_global.csv
```

## 11. Processing commands

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

## 12. Error handling

The system must stop for missing required columns, missing target values in final rows, nonnumeric target values, impossible horizon values, event overlap between splits, preprocessing artifact mismatch, model feature-order mismatch, corrupt TLE input, or unsupported dashboard artifacts.

Warnings are appropriate for insufficient 7-day coverage, truncation, missing optional feature groups, and unavailable calibration because a split contains too few positive events. Warnings must appear in reports.

## 13. Third-party integration

- Use [Python SGP4](https://github.com/brandon-rhodes/python-sgp4) for orbital support.
- Use [PyTorch-TCN](https://github.com/paul-krug/pytorch-tcn) or a documented local adaptation for TCN layers.
- Use [Kessler](https://github.com/kesslerlib/kessler) as a domain reference; address GPL-3.0 before copying code.
- Use [OrbVeil](https://github.com/ncdrone/orbveil) only as an optional Apache-2.0 parser/screening adapter after testing a pinned commit.

Record all versions and licenses in `THIRD_PARTY_NOTICES.md`.

## 14. Security and data handling

Never commit credentials, private test data, or downloaded archives. CelesTrak retrieval must have a timeout and user-agent. Validate remote JSON before parsing. Do not execute repository-provided scripts without review. Store only public data and derived artifacts in the project repository.
