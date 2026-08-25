# ORVEXA Canonical Project Structure

**Status:** Corrected structure after cross-checking the PRD, `AGENTS.md`, Technical Design, Testing Strategy, Research document, unified master plan, GitHub reuse guide, and vibe-coding specification.

## 1. Final decision

The earlier structure was substantially correct, but it had four inconsistencies:

1. `AGENTS.md` was not included in the original tree even though it is required for vibe coding.
2. The vibe-coding specification used `artifacts/preprocessors/`, but the other plans did not include that directory.
3. The unified master plan referred to `configs/features.yaml` and `requirements.txt`, while the technical design used `configs/base.yaml` and a lock file.
4. The technical documents described `src/orvexa/orbit/` but did not always list its internal files.

The structure below is now the single canonical structure. All future code should follow it.

## 2. Canonical tree

```text
orvexa/
├── README.md
├── LICENSE
├── CITATION.cff
├── THIRD_PARTY_NOTICES.md
├── AGENTS.md
├── ORVEXA_PRD.md
├── ORVEXA_TECHNICAL_DESIGN.md
├── ORVEXA_TESTING.md
├── ORVEXA_RESEARCH.md
├── ORVEXA_VIBE_CODING_SPEC.md
├── ORVEXA_CANONICAL_PROJECT_STRUCTURE.md
├── pyproject.toml
├── requirements-lock.txt
├── .env.example
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── features.yaml
│   ├── horizons.yaml
│   ├── models.yaml
│   └── experiments/
│       ├── A1_physics.yaml
│       ├── A2_linear.yaml
│       ├── A3_xgb_snapshot_no_mre.yaml
│       ├── A4_xgb_snapshot_mre.yaml
│       ├── A5_xgb_temporal_no_mre.yaml
│       ├── A6_xgb_temporal_mre.yaml
│       ├── A7_tcn_no_mre.yaml
│       ├── A8_tcn_mre.yaml
│       ├── A10_calibration.yaml
│       ├── A11_split_comparison.yaml
│       ├── A12_horizon_sweep.yaml
│       ├── A13_robustness.yaml
│       └── A14_explainability.yaml
│
├── data/
│   ├── README.md
│   ├── raw/                         # Git-ignored downloaded archives and source files
│   │   ├── esa/
│   │   ├── space_track/
│   │   └── celestrak/
│   ├── interim/                     # Git-ignored intermediate extracts
│   ├── processed/                   # Horizon and split datasets; Git-ignored by default
│   │   ├── event_views/
│   │   ├── sequences/
│   │   └── snapshots/
│   └── manifests/
│       ├── dataset_manifest.yaml
│       ├── schema_report.json
│       ├── split_manifest.json
│       └── retrieval_log.json
│
├── src/
│   └── orvexa/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data_io.py
│       ├── schema.py
│       ├── quality.py
│       ├── event_builder.py
│       ├── horizons.py
│       ├── splitting.py
│       ├── preprocessing.py
│       ├── features_snapshot.py
│       ├── features_temporal.py
│       ├── datasets.py
│       ├── models_base.py
│       ├── models_physics.py
│       ├── models_linear.py
│       ├── models_xgb.py
│       ├── models_tcn.py
│       ├── calibration.py
│       ├── ranking_metrics.py
│       ├── classification_metrics.py
│       ├── regression_metrics.py
│       ├── bootstrap.py
│       ├── robustness.py
│       ├── explainability.py
│       ├── reporting.py
│       ├── artifact_store.py
│       └── orbit/
│           ├── __init__.py
│           ├── tle_parser.py
│           ├── omm_parser.py
│           ├── validation.py
│           ├── propagation.py
│           └── coordinate_utils.py
│
├── scripts/
│   ├── download_data.py
│   ├── audit_data.py
│   ├── build_event_dataset.py
│   ├── make_splits.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── calibrate.py
│   ├── run_ablation.py
│   ├── run_robustness.py
│   ├── explain_model.py
│   ├── generate_figures.py
│   └── generate_report.py
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── mini_esa.csv
│   │   ├── sample_tle.txt
│   │   └── sample_omm.json
│   ├── test_schema.py
│   ├── test_quality.py
│   ├── test_event_builder.py
│   ├── test_horizon_filtering.py
│   ├── test_split_leakage.py
│   ├── test_preprocessing.py
│   ├── test_feature_registry.py
│   ├── test_models.py
│   ├── test_tcn_shapes.py
│   ├── test_metrics.py
│   ├── test_calibration.py
│   ├── test_bootstrap.py
│   ├── test_robustness.py
│   ├── test_artifacts.py
│   └── test_orbit_module.py
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_event_construction.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_tcn_training.ipynb
│   ├── 05_results.ipynb
│   └── 06_orbit_demo.ipynb
│
├── artifacts/                       # Git-ignored generated outputs
│   ├── models/
│   ├── preprocessors/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   ├── explanations/
│   └── logs/
│
├── reports/
│   ├── schema_report.json
│   ├── data_quality_report.html
│   ├── split_manifest.json
│   ├── experiment_summary.csv
│   └── reproducibility_report.md
│
└── app/
    ├── streamlit_app.py
    ├── components/
    │   ├── artifact_loader.py
    │   ├── filters.py
    │   ├── metric_cards.py
    │   ├── charts.py
    │   └── warnings.py
    ├── pages/
    │   ├── overview.py
    │   ├── ranked_alerts.py
    │   ├── event_detail.py
    │   ├── explanations.py
    │   ├── reliability.py
    │   ├── horizon_comparison.py
    │   ├── robustness.py
    │   └── orbital_demo.py
    └── assets/
```

## 3. Naming decisions

The canonical dependency file is `requirements-lock.txt`. If `uv` is used, add `uv.lock` as an additional generated lock file; do not replace the human-readable requirements lock without updating documentation.

The canonical shared configuration files are `base.yaml`, `features.yaml`, `horizons.yaml`, and `models.yaml`. `features.yaml` contains the feature groups, aliases, transforms, and with/without-MRE policy. Experiment-specific YAML files belong under `configs/experiments/`.

The canonical preprocessing output directory is `artifacts/preprocessors/`. This resolves the earlier omission and matches the artifact contract in the detailed vibe-coding specification.

Generated data and artifacts are not source code. They should be ignored by Git except for small illustrative fixtures, selected final figures, and explicitly approved summary outputs.

## 4. Responsibilities by layer

| Layer | Canonical location | Responsibility |
|---|---|---|
| Product requirements | `ORVEXA_PRD.md` | What is built and why. |
| AI behavior rules | `AGENTS.md` | How a coding assistant must work. |
| Scientific implementation contract | `ORVEXA_VIBE_CODING_SPEC.md` | Exact target, horizons, features, models, metrics, and invariants. |
| Technical architecture | `ORVEXA_TECHNICAL_DESIGN.md` | Modules, interfaces, artifacts, and data flow. |
| Testing | `ORVEXA_TESTING.md` | Verification, edge cases, leakage tests, and coverage. |
| Research context | `ORVEXA_RESEARCH.md` | Domain context, related work, risks, and novelty. |
| Configuration | `configs/` | No hard-coded experiment decisions. |
| Source code | `src/orvexa/` | Reusable production-style Python modules. |
| CLI wrappers | `scripts/` and `src/orvexa/cli.py` | Reproducible commands. |
| Tests | `tests/` | Correctness and scientific-validity checks. |
| Generated results | `artifacts/` and `reports/` | Models, metrics, figures, explanations, and audit outputs. |
| Demonstration | `app/` | Read-only Streamlit visualization. |

## 5. Minimal first scaffold

Create the full directories, but implement in this order:

```text
1. pyproject.toml, requirements-lock.txt, .gitignore, README.md, AGENTS.md
2. configs/base.yaml and configs/features.yaml
3. src/orvexa/config.py, schema.py, data_io.py, quality.py
4. event_builder.py, horizons.py, splitting.py
5. preprocessing.py, features_snapshot.py, features_temporal.py, datasets.py
6. models_physics.py, models_linear.py, models_xgb.py
7. models_tcn.py and test_tcn_shapes.py
8. metrics, bootstrap, calibration, reporting
9. explainability and robustness
10. Streamlit app
11. TLE/SGP4 supporting module
```

Do not begin with the dashboard. Do not begin with TLE integration. The first visible milestone is a reproducible ESA audit and leakage-free event dataset.

## 6. Final answer to the recheck

The previous structure was directionally correct but not fully synchronized. The tree in this document is the corrected canonical version. The important changes are the addition of `AGENTS.md`, `features.yaml`, `artifacts/preprocessors/`, explicit model/metric/bootstrap modules, explicit orbit submodules, experiment configuration files, and dashboard component/page separation.
