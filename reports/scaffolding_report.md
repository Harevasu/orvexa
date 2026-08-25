# ORVEXA Repository Scaffolding Report

**Date:** 2026-08-26  
**Status:** Completed Scaffolding Setup  
**Document Purpose:** Verification of repository structure, file tree creation, placeholder modules, testing discovery, and compliance with project invariants.

---

## 1. Project Invariants Compliance Verification

| Invariant / Constraint | Compliance Status | Evidence / Notes |
|---|---|---|
| **No ML Model Implementation or Training** | **CONFIRMED** | No models were fitted, initialized with weights, or trained. |
| **No Data Pipeline Execution** | **CONFIRMED** | No prefixes constructed; no transforms applied. |
| **No Dataset Modification** | **CONFIRMED** | Raw ESA CSV (`data/raw/esa/train_data.csv`) is completely untouched and hash-preserved. |
| **No External Data Downloaded** | **CONFIRMED** | No external requests were made; TLE/CelesTrak folders remain empty placeholders. |
| **No Outlawed Concepts Introduced** | **CONFIRMED** | GNN, RL, Computer Vision, debris reuse, circular economy, debris capture, and autonomous maneuvers are strictly omitted. |
| **No Dashboard Mock Predictions** | **CONFIRMED** | Streamlit UI contains only structural components with actionable empty states and disclaimers. |
| **No Documentation Rewritten** | **CONFIRMED** | All files in `docs/`, `rules.md`, and `ORVEXA_TESTING.md` remain intact as the original sources of truth. |

---

## 2. Directory Tree and Created Files

### A. Root Files
- `.gitignore`: Comprehensive git-ignore preventing accidental commits of raw/interim/processed datasets, model weights, cache, and virtual environments.
- `.env.example`: Environment variables template for paths and seeds.
- `pyproject.toml`: Standard Python build configuration with dependencies specified.
- `requirements-lock.txt`: Locked dependency requirements.
- `README.md`: Project overview, structure, installation instructions, and operational disclaimer.
- `LICENSE`: MIT License.
- `CITATION.cff`: Citation metadata for research attribution.
- `THIRD_PARTY_NOTICES.md`: Notices and licensing details for external software and datasets.

### B. Configuration Files (`configs/`)
- `configs/base.yaml`: Global configuration (horizons, data paths, seeds, split fractions, training defaults).
- `configs/features.yaml`: Feature whitelist, exclusions, and preprocessing transformations.
- `configs/horizons.yaml`: Warning cutoff horizons (2.0, 3.0, 5.0, 7.0 days).
- `configs/models.yaml`: Hyperparameter specifications for physics, linear, XGBoost, and TCN models.
- `configs/experiments/`:
  - `A1_physics.yaml`: ESA `max_risk_estimate` baseline.
  - `A2_linear.yaml`: Regularized Ridge and Logistic regression baselines.
  - `A3_xgb_snapshot_no_mre.yaml`: Snapshot XGBoost without `max_risk_estimate`.
  - `A4_xgb_snapshot_mre.yaml`: Snapshot XGBoost with `max_risk_estimate`.
  - `A5_xgb_temporal_no_mre.yaml`: Temporal-summary XGBoost without `max_risk_estimate`.
  - `A6_xgb_temporal_mre.yaml`: Temporal-summary XGBoost with `max_risk_estimate`.
  - `A7_tcn_no_mre.yaml`: Masked causal TCN without `max_risk_estimate`.
  - `A8_tcn_mre.yaml`: Masked causal TCN with `max_risk_estimate`.
  - `A10_calibration.yaml`: Platt and isotonic probability calibration.
  - `A11_split_comparison.yaml`: Chronological vs random event split evaluation.
  - `A12_horizon_sweep.yaml`: Multi-horizon (2, 3, 5, 7 day) performance comparison.
  - `A13_robustness.yaml`: Input perturbation (noise, missingness, sequence truncation) stress tests.
  - `A14_explainability.yaml`: Tree SHAP attributions and TCN temporal occlusion.

### C. Data Placeholders (`data/`)
- `data/README.md`: Data integrity guidelines and data-leakage prevention invariants.
- `data/raw/esa/train_data.csv`: Existing raw ESA training data (unmodified).
- `data/raw/space_track/.gitkeep`: Supporting TLE directory placeholder.
- `data/raw/celestrak/.gitkeep`: Supporting CelesTrak directory placeholder.
- `data/interim/.gitkeep`: Intermediate processing placeholder.
- `data/processed/event_views/.gitkeep`: Horizon-specific event views placeholder.
- `data/processed/sequences/.gitkeep`: Padded sequence tensors placeholder.
- `data/processed/snapshots/.gitkeep`: Latest qualifying snapshot feature tables placeholder.
- `data/manifests/.gitkeep`: Data manifests and schema reports placeholder.

### D. Source Code (`src/orvexa/`)
- `src/orvexa/__init__.py`: Package root with version metadata.
- `src/orvexa/cli.py`: CLI command dispatcher placeholder.
- `src/orvexa/config.py`: YAML configuration loader and validator placeholder.
- `src/orvexa/data_io.py`: Dataset loader and SHA-256 manifest manager placeholder.
- `src/orvexa/schema.py`: Schema registry and mandatory column validator placeholder.
- `src/orvexa/quality.py`: Missingness, duplicate, and quality audit placeholder.
- `src/orvexa/event_builder.py`: Conjunction event grouper and horizon prefix builder placeholder.
- `src/orvexa/horizons.py`: Warning horizon cutoff filtering definitions.
- `src/orvexa/splitting.py`: Chronological event-aware partition generator placeholder.
- `src/orvexa/preprocessing.py`: Train-only feature preprocessing transformer placeholder.
- `src/orvexa/features_snapshot.py`: Snapshot feature extractor placeholder.
- `src/orvexa/features_temporal.py`: Engineered temporal summary feature extractor placeholder.
- `src/orvexa/datasets.py`: PyTorch TCN sequence dataset and mask generator placeholder.
- `src/orvexa/models_base.py`: Abstract `RiskModel` protocol definition.
- `src/orvexa/models_physics.py`: Physics baseline estimator placeholder.
- `src/orvexa/models_linear.py`: Ridge / Logistic regression model placeholder.
- `src/orvexa/models_xgb.py`: Snapshot and temporal-summary XGBoost placeholder.
- `src/orvexa/models_tcn.py`: Masked causal TCN architecture placeholder.
- `src/orvexa/calibration.py`: Platt and isotonic calibrator placeholder.
- `src/orvexa/ranking_metrics.py`: Alert budget ranking metrics (Recall@K, Precision@K, NDCG@K) placeholder.
- `src/orvexa/classification_metrics.py`: PR-AUC, Brier score, and ECE placeholder.
- `src/orvexa/regression_metrics.py`: MAE, RMSE, Median AE, and Spearman correlation placeholder.
- `src/orvexa/bootstrap.py`: 2,000-iteration event-level bootstrap CI calculator placeholder.
- `src/orvexa/robustness.py`: Data corruption perturbation stress test placeholder.
- `src/orvexa/explainability.py`: SHAP and temporal occlusion explanation generator placeholder.
- `src/orvexa/reporting.py`: Metric collation and reliability figure generator placeholder.
- `src/orvexa/artifact_store.py`: Deterministic artifact path manager.
- `src/orvexa/orbit/__init__.py`: Orbital demonstration module root.
- `src/orvexa/orbit/tle_parser.py`: Two-Line Element (TLE) parser placeholder.
- `src/orvexa/orbit/omm_parser.py`: Orbit Mean-Elements Message (OMM) parser placeholder.
- `src/orvexa/orbit/validation.py`: Checksum and orbital element validation placeholder.
- `src/orvexa/orbit/propagation.py`: SGP4 propagation wrapper placeholder.
- `src/orvexa/orbit/coordinate_utils.py`: TEME / ECEF / Geodetic coordinate conversion placeholder.

### E. Pipeline Scripts (`scripts/`)
- `scripts/download_data.py`: Archive retrieval placeholder.
- `scripts/audit_data.py`: Schema & data quality audit runner placeholder.
- `scripts/build_event_dataset.py`: Event prefix dataset builder placeholder.
- `scripts/make_splits.py`: Chronological partition runner placeholder.
- `scripts/train_model.py`: Model training runner placeholder.
- `scripts/evaluate.py`: Evaluation metrics runner placeholder.
- `scripts/calibrate.py`: Calibration runner placeholder.
- `scripts/run_ablation.py`: Ablation study runner placeholder.
- `scripts/run_robustness.py`: Input perturbation runner placeholder.
- `scripts/explain_model.py`: Explainability runner placeholder.
- `scripts/generate_figures.py`: Thesis figure generation runner placeholder.
- `scripts/generate_report.py`: Reproducibility report generator placeholder.

### F. Tests and Fixtures (`tests/`)
- `tests/conftest.py`: Pytest / unittest fixtures configuration.
- `tests/fixtures/mini_esa.csv`: Synthetic mini CDM dataset fixture with valid column names.
- `tests/fixtures/sample_tle.txt`: Sample ISS TLE record for orbit parsing tests.
- `tests/fixtures/sample_omm.json`: Sample CCSDS OMM record for orbit parsing tests.
- `tests/test_schema.py`: Schema validation test scaffold.
- `tests/test_quality.py`: Data quality test scaffold.
- `tests/test_event_builder.py`: Event builder test scaffold.
- `tests/test_horizon_filtering.py`: Horizon cutoff filtering test scaffold.
- `tests/test_split_leakage.py`: Partition disjointness and leakage test scaffold.
- `tests/test_preprocessing.py`: Train-only transformer test scaffold.
- `tests/test_feature_registry.py`: Feature whitelist test scaffold.
- `tests/test_models.py`: Model protocol test scaffold.
- `tests/test_tcn_shapes.py`: TCN tensor shape and mask invariant test scaffold.
- `tests/test_metrics.py`: Metrics test scaffold.
- `tests/test_calibration.py`: Probability calibration test scaffold.
- `tests/test_bootstrap.py`: Event bootstrap test scaffold.
- `tests/test_robustness.py`: Robustness perturbation test scaffold.
- `tests/test_artifacts.py`: Artifact store test scaffold.
- `tests/test_orbit_module.py`: Orbit parsing and SGP4 propagation test scaffold.

### G. Jupyter Notebooks (`notebooks/`)
- `notebooks/01_data_audit.ipynb`: Data audit exploration notebook.
- `notebooks/02_event_construction.ipynb`: Event construction exploration notebook.
- `notebooks/03_baselines.ipynb`: Baseline model exploration notebook.
- `notebooks/04_tcn_training.ipynb`: TCN model training notebook.
- `notebooks/05_results.ipynb`: Results analysis and explainability notebook.
- `notebooks/06_orbit_demo.ipynb`: SGP4 orbital propagation demo notebook.

### H. Generated Artifact and Model Directories
- `artifacts/models/.gitkeep`
- `artifacts/preprocessors/.gitkeep`
- `artifacts/predictions/.gitkeep`
- `artifacts/metrics/.gitkeep`
- `artifacts/figures/.gitkeep`
- `artifacts/explanations/.gitkeep`
- `artifacts/logs/.gitkeep`
- `models/.gitkeep`

### I. Streamlit Application (`app/`)
- `app/streamlit_app.py`: Main entry point with research disclaimer and page routing.
- `app/components/__init__.py`: Component package root.
- `app/components/artifact_loader.py`: Safe artifact loader with empty state handlers.
- `app/components/filters.py`: UI filter controls.
- `app/components/metric_cards.py`: Metric summary cards.
- `app/components/charts.py`: Chart rendering utilities.
- `app/components/warnings.py`: Operational limitation disclaimer component.
- `app/pages/__init__.py`: Pages package root.
- `app/pages/overview.py`: Project overview page.
- `app/pages/ranked_alerts.py`: Ranked conjunction alerts page.
- `app/pages/event_detail.py`: Conjunction event detail page.
- `app/pages/explanations.py`: Model explanation page.
- `app/pages/reliability.py`: Probability calibration page.
- `app/pages/horizon_comparison.py`: Multi-horizon comparison page.
- `app/pages/robustness.py`: Degraded data robustness page.
- `app/pages/orbital_demo.py`: SGP4 orbit demonstration page.
- `app/assets/.gitkeep`: Dashboard assets directory.

---

## 3. Existing Files Not Modified

The following files pre-existed in the repository and were preserved without modification:
1. `rules.md`
2. `ORVEXA_TESTING.md`
3. `data/raw/esa/train_data.csv`
4. `docs/DOCUMENTATION_AUDIT.md`
5. `docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md`
6. `docs/ORVEXA_PRD.md`
7. `docs/ORVEXA_RESEARCH.md`
8. `docs/ORVEXA_TECHNICAL_DESIGN.md`
9. `docs/ORVEXA_VIBE_CODING_SPEC.md`

---

## 4. Validation Performed

1. **Package Importability Test:**
   - Executed dynamic package walkthrough importing all 32 modules under `orvexa` and `orvexa.orbit`.
   - Result: All modules imported cleanly with zero syntax or import errors.
2. **Test Discovery Execution:**
   - Executed `python -m unittest discover tests`.
   - Result: 15/15 test modules successfully discovered and executed with 100% pass rate (`OK`).
3. **Git Hygiene & Ignore Rules Check:**
   - Executed `git status --ignored`.
   - Result: `data/raw/esa/train_data.csv` (233 MB) is properly ignored and protected from accidental staging; all scaffolding files are properly organized.

---

## 5. Unresolved Documentation Issues

No blocking contradictions remain. As noted in the documentation audit:
- Experiment `A9` is an analytical comparison of the artifacts generated by `A3-A8` rather than an independently trained model, so it relies on base and experiment configurations without requiring a standalone training config.
- External TLE download is deferred until core ML milestones are complete.

---

## 6. Next Step

The repository scaffolding is complete, verified, and ready. The next planned task is the **ESA CSV Schema and Data Audit** (`audit-data`).
