# ORVEXA

> **An Explainable and Calibrated Temporal AI Framework for Multi-Horizon Spacecraft Conjunction-Risk Prioritization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Overview

**ORVEXA** is a research decision-support prototype that evaluates whether a masked causal Temporal Convolutional Network (TCN) improves event-level spacecraft conjunction-risk prioritization over physics-derived estimates, snapshot XGBoost, and temporal-summary XGBoost models using historical European Space Agency (ESA) Conjunction Data Messages (CDMs).

> **Important Operational Disclaimer:**  
> **Research estimate only. ORVEXA is not an operational collision-avoidance authority.**  
> ORVEXA does not control spacecraft, plan maneuvers, perform debris removal, or certify collision probabilities.

---

## 2. Repository Structure

```text
orvexa/
├── rules.md                           # AI context and behavioral rules (Priority 1)
├── ORVEXA_TESTING.md                  # Testing and verification strategy
├── pyproject.toml                     # Python package configuration
├── requirements-lock.txt              # Frozen dependency specifications
├── .env.example                       # Environment variables template
├── .gitignore                         # Git exclusion rules
│
├── configs/                           # Experiment & pipeline configurations
│   ├── base.yaml                      # Global configuration
│   ├── features.yaml                  # Feature registry and group whitelist
│   ├── horizons.yaml                  # Warning horizons (2, 3, 5, 7 days)
│   ├── models.yaml                    # Model family specifications
│   └── experiments/                   # Experiment A1-A14 configs
│
├── data/                              # Data storage
│   ├── README.md                      # Data handling rules & integrity guidelines
│   ├── raw/esa/                       # Original ESA train_data.csv (immutable)
│   ├── interim/                       # Intermediate artifacts
│   ├── processed/                     # Leakage-free event views and sequences
│   └── manifests/                     # Data checksums and schema reports
│
├── src/orvexa/                        # Core Python package
│   ├── cli.py                         # Command-line interface
│   ├── config.py                      # YAML config validator
│   ├── data_io.py                     # Data loaders & manifest utilities
│   ├── schema.py                      # Schema verification & feature whitelist
│   ├── quality.py                     # Data quality and missingness auditing
│   ├── event_builder.py               # Event grouping and horizon prefix construction
│   ├── horizons.py                    # Horizon cutoff definitions
│   ├── splitting.py                   # Event-aware chronological splitting
│   ├── preprocessing.py               # Train-only imputation and encoding
│   ├── features_snapshot.py           # Snapshot feature extractor
│   ├── features_temporal.py           # Temporal summary extractor
│   ├── datasets.py                    # PyTorch TCN sequence dataset & DataLoader
│   ├── models_base.py                 # Abstract base model protocols
│   ├── models_physics.py              # Max risk estimate baseline
│   ├── models_linear.py               # Regularized linear baselines
│   ├── models_xgb.py                  # Snapshot & temporal XGBoost
│   ├── models_tcn.py                  # Masked causal TCN architecture
│   ├── calibration.py                 # Platt & isotonic calibrators
│   ├── ranking_metrics.py             # Recall@K, Precision@K, NDCG@K
│   ├── classification_metrics.py      # PR-AUC, Brier, ECE
│   ├── regression_metrics.py          # MAE, RMSE, Spearman
│   ├── bootstrap.py                   # Event-level bootstrap confidence intervals
│   ├── robustness.py                  # Data perturbation & stress testing
│   ├── explainability.py              # Tree SHAP and TCN temporal occlusion
│   ├── reporting.py                   # Table and figure generators
│   ├── artifact_store.py              # Saved model and metric management
│   └── orbit/                         # Auxiliary SGP4 TLE/OMM propagation module
│
├── scripts/                           # Reproducible pipeline scripts
├── tests/                             # Test suites and fixtures
├── artifacts/                         # Generated models, metrics, figures (git-ignored)
├── reports/                           # Audit and summary reports
├── docs/                              # Project specifications and PRD
└── app/                               # Streamlit read-only dashboard
```

---

## 3. Quickstart

### Installation

```bash
# Clone repository
git clone https://github.com/orvexa/orvexa.git
cd orvexa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Running Verification Tests

```bash
pytest
```

---

## 4. Documentation

Detailed specifications and architectural documents are located in `docs/`:
- [`ORVEXA_PRD.md`](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_PRD.md)
- [`ORVEXA_TECHNICAL_DESIGN.md`](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_TECHNICAL_DESIGN.md)
- [`ORVEXA_VIBE_CODING_SPEC.md`](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_VIBE_CODING_SPEC.md)
- [`ORVEXA_CANONICAL_PROJECT_STRUCTURE.md`](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md)
- [`rules.md`](file:///c:/Users/Zyren/Documents/orvexa/rules.md)
- [`ORVEXA_TESTING.md`](file:///c:/Users/Zyren/Documents/orvexa/ORVEXA_TESTING.md)
