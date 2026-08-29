# ORVEXA

> **An Explainable and Calibrated Temporal AI Framework for Multi-Horizon Spacecraft Conjunction-Risk Prioritization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3+-61DAFB.svg?logo=react&logoColor=black)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-6.2+-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 158 Passed](https://img.shields.io/badge/Tests-158%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Data: ESA Challenge](https://img.shields.io/badge/Dataset-ESA%20Kelvins%20CDMs-blue.svg)](https://kelvins.esa.int/collision-avoidance-challenge/)

---

## 1. Overview

**ORVEXA** is a research decision-support prototype that evaluates whether deep causal sequence modeling and distribution-free conformal uncertainty quantification improve event-level spacecraft conjunction-risk prioritization over physics-derived estimates, snapshot tabular models, and temporal-summary XGBoost baselines using historical European Space Agency (ESA) Conjunction Data Messages (CDMs).

Spacecraft conjunction assessment is intrinsically sequential: as the Time of Closest Approach (TCA) approaches, tracking radars and optical sensors acquire new observations, updating orbit ephemerides and covariance ellipses. ORVEXA models these evolving conjunction time series using **Masked Causal Temporal Convolutional Networks (TCNs)** and **Conformalized Quantile Regression (CQR)** across multiple operational warning horizons ($H2$, $H3$, $H5$, and $H6$).

```
                                      ORVEXA Pipeline Architecture
 ┌─────────────────┐      ┌─────────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
 │ Historical CDMs │ ───► │  Horizon Prefix Filter  │ ───► │  Masked Causal TCN (M4)   │ ───► │  Conformal Quantile     │
 │ (162k rows,     │      │  (Strict t >= H cutoff; │      │  - Dilated Causal Conv1D  │      │  Regression (CQR)       │
 │  13.1k events)  │      │   Zero future leakage)  │      │  - Non-crossing Quantiles │      │  - Valid 90% Coverage   │
 └─────────────────┘      └─────────────────────────┘      └───────────────────────────┘      └─────────────────────────┘
                                                                                                           │
                                                                                                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐   ┌─────────────────────────┐
 │                       ORVEXA Demonstration Web Platform                                │   │ Calibrated Triage Alert │
 │  - React + TypeScript + Vite Space Situational Awareness (SSA) Control Dashboard       │◄──│  - Point Estimate (q50) │
 │  - FastAPI Read-Only Inference Engine Backed by Frozen Candidate C Artifacts           │   │  - Adaptive [q05, q95]  │
 └────────────────────────────────────────────────────────────────────────────────────────┘   │  - Ranked Alert Queue   │
                                                                                              └─────────────────────────┘
```

> [!IMPORTANT]
> **Operational Safety Disclaimer:**  
> **Research estimate only. ORVEXA is not an operational collision-avoidance authority.**  
> ORVEXA does not control spacecraft, plan maneuvers, perform debris removal, or certify collision probabilities.

---

## 2. Key Scientific Findings & Benchmark Results

ORVEXA has been evaluated across five rigorous experimental phases adhering to strict chronological event-level splitting, train-only preprocessor fitting, and cryptographic artifact freezing.

### 2.1 The Three Operational Regimes

The empirical trajectory across lead times reveals three distinct physical regimes in spacecraft conjunction assessment:

1. **Regime 1: Tight Deterministic Control ($H \le 48\text{h}$ / $H2$)**  
   High observation density yields tight covariance constraints. Continuous point regression is highly accurate ($R^2 = +0.52 \text{ to } +0.58$, Spearman $\rho \approx 0.72$), capturing $\ge 90\%$ of critical events within a 5% operational alert budget.
2. **Regime 2: Degraded Point Signal / Viable Triage ($48\text{h} < H \le 120\text{h}$ / $H3\text{--}H5$)**  
   Atmospheric drag and orbit propagation dispersion widen the error distribution ($R^2 = +0.39 \to +0.19$). While exact point accuracy degrades, monotonic rank ordering remains strong ($\rho \approx 0.52 - 0.68$), enabling effective early-warning screening.
3. **Regime 3: Continuous Regression Failure Boundary ($H \ge 144\text{h}$ / $H6$)**  
   At 6 days prior to TCA, deterministic continuous regression fails ($R^2 = -0.016 \text{ to } -0.167$) because physical orbital uncertainties dominate. In this regime, **CQR Conformal Prediction provides the only mathematically defensible tool**, delivering a verified **90.20% marginal coverage** and **100% tail coverage** ($y \ge -5.0$).

---

### 2.2 Authoritative Historical Benchmark (Phase 3B & Phase 4A)

Evaluated on the permanently sealed **Historical Master Test Partition** ($N = 1,974$ events, indices $[11180, 13153]$) using deterministic candidate model $M_4$ (37 channels: One-Hot Categoricals + $\log_{10}$ Covariances):

| Horizon ($H$) | Lead Time | Qualifying Events ($N$) | MAE | RMSE | $R^2$ | Pearson $r$ | Spearman $\rho$ | Critical Events ($y \ge -5.0$) | Recall @ 5% Budget | Recall @ 10% Budget | Precision @ 10% Budget |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H2$** | 48.0h (2.0d) | 1,799 | **3.3257** | **6.7579** | **+0.5163** | 0.7475 | 0.6772 | 10 | **90.0%** (9/10) | **90.0%** (9/10) | 5.03% (9/179) |
| **$H3$** | 72.0h (3.0d) | 1,700 | **3.9943** | **7.6320** | **+0.3876** | 0.6760 | 0.6430 | 8 | **37.5%** (3/8) | **62.5%** (5/8) | 2.94% (5/170) |
| **$H5$** | 120.0h (5.0d) | 1,437 | **5.1215** | **8.9892** | **+0.1413** | 0.5198 | 0.5012 | 7 | **14.3%** (1/7) | **14.3%** (1/7) | 0.70% (1/143) |
| **$H6$** | 144.0h (6.0d) | 1,279 | **5.6799** | **9.6195** | **-0.0162** | 0.4149 | 0.4022 | 6 | **0.0%** (0/6) | **16.7%** (1/6) | 0.79% (1/127) |

---

### 2.3 Phase 5 Probabilistic Benchmark (Quantile M4 + CQR)

Evaluated on the independent **Phase 5 Internal Test Partition** ($N = 1,677$ events, indices $[9503, 11179]$) with 95% bootstrap confidence intervals (1,000 resamples):

| Horizon ($H$) | Lead Time | Test Samples ($N$) | Point Pred $\hat{q}_{0.50}$ MAE | Point Pred $\hat{q}_{0.50}$ $R^2$ [95% CI] | Spearman $\rho$ | Mean Pinball Loss | CQR 90% Empirical Coverage | CQR Mean Interval Width | CQR Tail Coverage ($y \ge -5.0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H2$** | 48.0h (2.0d) | 1,528 | **3.3418** | **+0.5850** [0.535, 0.639] | **0.7188** | 0.9869 | **92.15%** | **12.54** | 75.0% (3/4) |
| **$H3$** | 72.0h (3.0d) | 1,429 | **3.7289** | **+0.4842** [0.425, 0.546] | **0.6802** | 1.1688 | **92.58%** | **17.95** | 66.7% (2/3) |
| **$H5$** | 120.0h (5.0d) | 1,193 | **5.3404** | **+0.1897** [0.124, 0.260] | **0.5228** | 1.4805 | **92.96%** | **20.83** | **100.0%** (3/3) |
| **$H6$** | 144.0h (6.0d) | 1,071 | **5.9731** | **-0.1665** [-0.245, -0.087] | **0.3785** | 1.5580 | **90.20%** | **18.03** | **100.0%** (3/3) |

> [!NOTE]
> - **Zero Crossing Violations:** Quantile heads guarantee strict monotonicity ($\hat{q}_{0.05} < \hat{q}_{0.50} < \hat{q}_{0.95}$) via non-negative softplus increments ($0.000000$ crossing rate across all validation sequences).
> - **Adaptive Sharpness:** Conformalized Quantile Regression (CQR) reduces uncertainty interval widths by **47.1% to 52.5%** relative to standard constant-width residual conformal prediction.

---

## 3. Methodological Architecture

### 3.1 Strict Anti-Leakage Chronological Governance

To prevent data contamination and temporal lookahead bias, ORVEXA enforces strict event-level chronological partitioning and prefix filtering:

```
Total Conjunction Event Pool (13,154 Unique Events)
  ├── Eligible Phase 5 Events (11,180 Events / 85.0%)
  │     ├── Phase 5 Train:        6,708 Events (60.0%) [Indices 0 to 6707]      ──► Normalizer fitting & model training ONLY
  │     ├── Phase 5 Validation:   1,677 Events (15.0%) [Indices 6708 to 8384]   ──► Early stopping & candidate audit ONLY
  │     ├── Phase 5 Calibration:  1,118 Events (10.0%) [Indices 8385 to 9502]   ──► Conformal quantile calibration ONLY
  │     └── Phase 5 Internal Test:1,677 Events (15.0%) [Indices 9503 to 11179]  ──► Blind final evaluation ONLY
  │
  └── Historical Master Test (1,974 Events / 15.0%)    [Indices 11180 to 13153] ──► PERMANENTLY QUARANTINED
```

- **Horizon Cutoff Adherence:** At warning horizon $H$, all CDM records with $\text{time\_to\_tca} < H$ are excluded.
- **Train-Only Preprocessing:** Imputation statistics, min-max scaling, and categorical encoders are fitted strictly on the training partition and applied out-of-sample to validation, calibration, and test partitions.
- **Sequence Contract:** Time series are left-padded to sequence length $L=23$ with validity masks ($1 = \text{valid}$, $0 = \text{padding}$). Pooling extracts representations strictly from the final valid timestep.

### 3.2 Model Families

1. **Physics Baselines:** `max_risk_estimate` (ESA CDMs maximum operational risk proxy) and `max_risk_scaling`.
2. **Tabular Baselines:** Ridge Regression, Snapshot XGBoost (most recent permissible CDM), and Temporal-Summary XGBoost (mean, std, min, max, delta over the CDM sequence).
3. **Deterministic Causal TCN ($M_0 \to M_5$):** Multi-layer causal 1D dilated convolutions with receptive field covering full CDM histories, Huber loss ($\delta=1.0$), AdamW optimizer, and Cosine Annealing.
4. **Quantile Causal TCN + CQR:** Multi-quantile pinball loss for $q \in \{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95\}$ with non-crossing parameterization:
   $$\hat{q}_{0.05}(x) = f_{0.05}(x)$$
   $$\hat{q}_{0.50}(x) = \hat{q}_{0.05}(x) + \text{softplus}(f_{\Delta 1}(x)) + \epsilon$$
   $$\hat{q}_{0.95}(x) = \hat{q}_{0.50}(x) + \text{softplus}(f_{\Delta 2}(x)) + \epsilon$$
   Calibrated via split Conformalized Quantile Regression (CQR) to guarantee finite-sample marginal coverage $1 - \alpha = 0.90$.

---

## 4. Repository Layout

```text
orvexa/
├── README.md                           # Master project documentation (this file)
├── pyproject.toml                     # Python package specifications and dependencies
├── requirements-lock.txt              # Frozen, reproducible dependency tree
├── rules.md                           # Context and engineering invariants
├── ORVEXA_TESTING.md                  # Verification suite protocol
│
├── backend/                           # FastAPI read-only inference backend
│   ├── main.py                        # API routes, CORS middleware, startup audit
│   ├── schemas.py                     # Pydantic data schemas
│   ├── inference.py                   # Read-only Candidate C inference dispatcher
│   ├── data_service.py                # Safe non-sealed demo event loader & catalog
│   ├── benchmarks_service.py          # Scientific benchmark data service
│   └── orbital_service.py             # Auxiliary SGP4 orbit ephemeris engine
│
├── frontend/                          # Space Situational Awareness React frontend
│   ├── package.json                   # NPM configuration & dependencies
│   ├── vite.config.ts                 # Vite bundler configuration & backend proxy
│   ├── tailwind.config.js             # Aerospace dark theme styling
│   └── src/
│       ├── App.tsx                    # Main layout and tab state manager
│       ├── components/                # Header, Sidebar, MetricCard, QuantileChart
│       ├── pages/                     # 8 dedicated demonstration views
│       ├── services/api.ts            # Typed HTTP API client
│       └── types/api.ts               # TypeScript data interfaces
│
├── configs/                           # Experiment and pipeline configurations
│   ├── base.yaml                      # Global configuration
│   ├── features.yaml                  # Feature registry and group whitelists
│   ├── horizons.yaml                  # Warning horizon cutoffs (H2, H3, H5, H6)
│   ├── models.yaml                    # Model hyperparameter specifications
│   └── phase3b_experiments.json       # Feature intervention configurations (M0-M5)
│
├── data/                              # Data storage (Git-ignored raw assets)
│   ├── README.md                      # Data handling rules & integrity guidelines
│   ├── raw/esa/                       # Original ESA train_data.csv (immutable)
│   ├── processed/                     # Leakage-free event views, sequences, snapshots
│   └── manifests/                     # Checksum manifests and schema reports
│
├── src/orvexa/                        # Core Python package
│   ├── models_tcn.py                  # Masked causal TCN architecture
│   ├── models_probabilistic.py        # Non-crossing Quantile TCN architecture
│   ├── conformal.py                   # Split conformal & CQR calibrators
│   ├── losses_quantile.py             # Multi-quantile pinball loss functions
│   ├── preprocessing_phase3b.py       # Advanced M4 transform preprocessor
│   ├── regression_metrics.py          # MAE, RMSE, Pearson r, Spearman rho, R²
│   ├── ranking_metrics.py             # Recall@K, Precision@K, NDCG@K
│   ├── bootstrap.py                   # Event-level bootstrap confidence intervals
│   └── orbit/                         # Auxiliary SGP4 TLE/OMM propagation module
│
├── scripts/                           # Reproducible pipeline execution scripts
├── tests/                             # Comprehensive test suite (158 passed tests)
├── artifacts/                         # Frozen candidate models, calibrators, preprocessors
├── reports/                           # Formal scientific audits and benchmark reports
├── docs/                              # Architecture, PRD, and technical specifications
└── app/                               # Auxiliary Streamlit research dashboard
```

---

## 5. Quickstart & Demonstration Guide

### 5.1 Environment Setup

ORVEXA requires **Python 3.11+** and **Node.js 18+**.

```bash
# 1. Clone the repository
git clone https://github.com/orvexa/orvexa.git
cd orvexa

# 2. Create and activate a Python virtual environment
python -m venv .venv-orvexa
source .venv-orvexa/bin/activate       # On Linux / macOS
# .venv-orvexa\Scripts\activate        # On Windows (PowerShell / CMD)

# 3. Install Python package and dependencies
pip install -e ".[dev]"
pip install fastapi httpx
```

---

### 5.2 Running the Web Demonstration Platform

The primary ORVEXA demonstration platform consists of a **FastAPI backend** (running real-time inference on frozen Candidate C weights) paired with a modern **React + Vite + TypeScript frontend**:

#### Terminal 1: Launch FastAPI Backend
```powershell
.venv-orvexa\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Terminal 2: Launch React Frontend
```powershell
cd frontend
npm install
npm run dev
```

Open your browser at: **`http://localhost:5173`**

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ORVEXA Web Demonstration Modules                │
├────────────────────────────┬───────────────────────────────────────────┤
│ 1. Conjunction Risk        │ Interactive workbench: select event &     │
│    Analysis (Main)         │ horizon, click "Analyze Conjunction",     │
│                            │ view median q50 & 90% CQR uncertainty.   │
├────────────────────────────┼───────────────────────────────────────────┤
│ 2. Event Detail & CDMs     │ Chronological CDM approach sequence,      │
│                            │ miss distance & Mahalanobis evolution.    │
├────────────────────────────┼───────────────────────────────────────────┤
│ 3. Horizon Comparison      │ Side-by-side sweep across H2, H3, H5, H6  │
│                            │ with point R² vs CQR coverage trade-off.  │
├────────────────────────────┼───────────────────────────────────────────┤
│ 4. Ranked Alerts Queue     │ Prioritized triage queue sorted by risk   │
│                            │ with one-click workbench loading.         │
├────────────────────────────┼───────────────────────────────────────────┤
│ 5. Reliability Audit       │ Empirical coverage verification on 1,677  │
│                            │ test events (90.20% - 92.96% coverage).   │
├────────────────────────────┼───────────────────────────────────────────┤
│ 6. Robustness & Limits     │ Unmasked H6 negative R² (-0.16651) and    │
│                            │ scientific explanation of uncertainty.   │
├────────────────────────────┼───────────────────────────────────────────┤
│ 7. Scientific Glossary     │ Technical guide to CDMs, TCA, log10(Pc),  │
│                            │ Causal TCNs, Pinball loss, and CQR.       │
├────────────────────────────┼───────────────────────────────────────────┤
│ 8. Auxiliary Ephemeris     │ SGP4 orbit propagation visualizer         │
│                            │ decoupled from ML risk scoring.           │
└────────────────────────────┴───────────────────────────────────────────┘
```

---

### 5.3 Running the Test Suite

Verify complete repository integrity, backend API endpoints, and scientific contracts:

```bash
# Run full test suite (158 tests)
pytest -v

# Run backend API integration tests only
pytest tests/test_backend_api.py -v

# Run frontend production build & TypeScript typecheck
cd frontend
npm run build
```

*Expected: 158 passed Python tests (100%) and clean TypeScript bundle compilation.*

---

### 5.4 End-to-End Pipeline Reproduction

To reproduce the multi-phase experimental benchmarks from scratch:

```bash
# 1. Audit raw ESA dataset and generate schema report
python scripts/audit_data.py

# 2. Construct horizon-filtered event views and sequence tensors (H2, H3, H5, H6)
python scripts/build_event_dataset.py
python scripts/build_h6_dataset.py

# 3. Train and evaluate Phase 1 tabular baselines
python scripts/run_phase1_baselines.py

# 4. Train and evaluate Phase 2B causal TCN sequence models
python scripts/run_phase2b_temporal.py

# 5. Execute Phase 3B feature intervention study (M0 through M5)
python scripts/run_phase3b_combined_experiments.py

# 6. Train and evaluate Phase 4A 6-day horizon extension (H6)
python scripts/run_phase4a_step2_training.py

# 7. Train Phase 5 Quantile M4 TCN models (H2, H3, H5, H6)
python scripts/run_phase5_step3_probabilistic_training.py

# 8. Fit CQR Conformal Calibrators and audit validation set
python scripts/run_phase5_step4_audit.py

# 9. Evaluate blind Phase 5 Internal Test set
python scripts/run_phase5_step5_blind_internal_test.py

# 10. Execute final scientific audit and cross-phase reconciliation
python scripts/run_phase5_step6_final_scientific_audit.py
```

---

## 6. Scientific Documentation & Audit Trail

All experimental phases are documented in formal scientific audit reports:

| Phase | Milestone / Document | Focus Area |
| :---: | :--- | :--- |
| **Design** | [`docs/ORVEXA_PRD.md`](docs/ORVEXA_PRD.md) | Product Requirements Document |
| **Design** | [`docs/ORVEXA_TECHNICAL_DESIGN.md`](docs/ORVEXA_TECHNICAL_DESIGN.md) | Technical Architecture & Equations |
| **Phase 2A** | [`reports/PHASE_2A_TEMPORAL_AUDIT.md`](reports/PHASE_2A_TEMPORAL_AUDIT.md) | Temporal Leakage Audit & Protocol Freeze |
| **Phase 2B** | [`reports/PHASE_2B_TEMPORAL_TRAINING_REPORT.md`](reports/PHASE_2B_TEMPORAL_TRAINING_REPORT.md) | Causal TCN Sequential Modeling Benchmark |
| **Phase 3A** | [`reports/PHASE_3A_TEMPORAL_DIAGNOSTIC_REPORT.md`](reports/PHASE_3A_TEMPORAL_DIAGNOSTIC_REPORT.md) | Diagnostic Failure Mode & Tail Analysis |
| **Phase 3B** | [`reports/PHASE_3B_FINAL_SCIENTIFIC_AUDIT.md`](reports/PHASE_3B_FINAL_SCIENTIFIC_AUDIT.md) | Feature Intervention Study & Blind Test ($M_0 \to M_5$) |
| **Phase 4A** | [`reports/PHASE_4A_STEP5_SCIENTIFIC_RECORD_CORRECTION.md`](reports/PHASE_4A_STEP5_SCIENTIFIC_RECORD_CORRECTION.md) | 6-Day Horizon Extension ($H6$) & Erratum Resolution |
| **Phase 5** | [`reports/PHASE_5_STEP6_FINAL_SCIENTIFIC_AUDIT.md`](reports/PHASE_5_STEP6_FINAL_SCIENTIFIC_AUDIT.md) | Quantile TCNs, Conformal Prediction (CQR), & Final Reconciliation |

---

## 7. Citation & Acknowledgments

If you use ORVEXA in your academic research or space situational awareness benchmarking, please cite:

```bibtex
@misc{orvexa2026,
  author = {ORVEXA Project Team},
  title = {ORVEXA: An Explainable and Calibrated Temporal AI Framework for Multi-Horizon Spacecraft Conjunction-Risk Prioritization},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/orvexa/orvexa}},
  note = {Research Decision-Support Prototype}
}
```

### Acknowledgments & Data Sources
- **European Space Agency (ESA):** Space Debris Office and the Kelvins Collision Avoidance Challenge dataset.
- **Space-Track & CelesTrak:** Auxiliary orbital mechanics and TLE reference ephemerides.
- **PyTorch-TCN, FastAPI, React & scikit-learn:** Core sequence modeling, API, and frontend presentation frameworks.

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
