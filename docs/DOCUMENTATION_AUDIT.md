# ORVEXA Documentation Consistency Audit Report

**Date:** 2026-08-26  
**Status:** Complete Documentation Audit  
**Target Document:** `docs/DOCUMENTATION_AUDIT.md`  

---

## A. Files Inspected

The following documents were inspected in strict accordance with the defined priority hierarchy:

1. [rules.md](file:///c:/Users/Zyren/Documents/orvexa/rules.md) (Priority 1 — Primary behavioral and architectural rulebook)
2. [docs/ORVEXA_VIBE_CODING_SPEC.md](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_VIBE_CODING_SPEC.md) (Priority 2 — Frozen scientific implementation contract)
3. [docs/ORVEXA_TECHNICAL_DESIGN.md](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_TECHNICAL_DESIGN.md) (Priority 3 — System architecture, data flow, interfaces)
4. [docs/ORVEXA_PRD.md](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_PRD.md) (Priority 4 — Product requirements, goals, user flows, acceptance criteria)
5. [docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md) (Priority 5 — Directory tree and module layout specification)
6. [ORVEXA_TESTING.md](file:///c:/Users/Zyren/Documents/orvexa/ORVEXA_TESTING.md) (Priority 6 — Testing strategy, test suites, invariants, assertions)
7. [docs/ORVEXA_RESEARCH.md](file:///c:/Users/Zyren/Documents/orvexa/docs/ORVEXA_RESEARCH.md) (Priority 7 — Academic/industry context, benchmarks, risk register)

---

## B. Consistent Requirements

The core scientific, architectural, and methodological foundations across all documents are coherent and aligned:

1. **Problem Formulation & Core Objective:**
   - Event-level spacecraft conjunction risk prioritization using historical ESA Conjunction Data Messages (CDMs).
   - Formulation as a multi-horizon warning task ($H \in \{2.0, 3.0, 5.0, 7.0\}$ days before Time of Closest Approach, TCA).
   - Core research question: Evaluating whether a masked causal Temporal Convolutional Network (TCN) provides tangible operational decision-support value over physics baselines (`max_risk_estimate`), regularized linear models, snapshot XGBoost, and temporal-summary XGBoost.

2. **Primary Supervised Target:**
   - The target is strictly the final CDM row's `risk` value within each conjunction event (`event_id`).
   - The target is treated as the **final ESA base-10 log-risk score**, never as a directly observed physical collision probability.
   - Regression uses identity transformation with Huber loss ($\delta = 1.0$).
   - Secondary classification uses sensitivity analysis across fixed log-risk thresholds $[-6.0, -5.0, -4.0]$ ($risk \ge \text{threshold}_{\log_{10}}$).

3. **Data Partitioning & Leakage Prevention:**
   - 100% disjoint event-level partitioning: 70% train, 15% validation, 15% test.
   - Splitting is chronological/ordered (by validated CDM creation timestamp or deterministic raw first-appearance proxy); random row splitting is strictly prohibited.
   - Preprocessing transformers (imputers, scalers, encoders) and calibration maps are fitted strictly on training (or validation for calibration) partitions.
   - Strict prefix horizon filtering: For horizon $H$, all input features must satisfy $\text{time\_to\_tca} \ge H$. Post-cutoff CDMs and final CDMs are strictly excluded from input features.

4. **TCN Architecture & Hyperparameters:**
   - Masked causal TCN with left-padding to max sequence length $T=23$ (based on measured max event sequence length).
   - Hyperparameter defaults: channels `[64, 64, 128]`, kernel size `3`, dilations `[1, 2, 4]`, causal convolution, dropout `0.15`, optimizer `AdamW` ($\text{lr}=10^{-3}, \text{weight\_decay}=10^{-4}$), batch size `64`, max epochs `100`, early stopping patience `12`, loss `Huber(delta=1.0)`.
   - Readout uses masked mean pooling over valid timesteps only (`mask=1` for real CDMs, `0` for padding).

5. **Operational Evaluation Framework:**
   - Headline metric: Recall@$K$ ($K \in \{1\%, 5\%, 10\%\}$) under fixed event alert budgets.
   - Complementary metrics: MAE, RMSE, median absolute error, Spearman correlation, PR-AUC, Precision@$K$, NDCG@$K$, Brier score, Expected Calibration Error (ECE), and 2,000-iteration event-level bootstrap confidence intervals (seed 42).

6. **Orbital Data Decoupling:**
   - Space-Track TLE History and CelesTrak data are isolated as auxiliary orbital propagation/visualization datasets.
   - Arbitrary TLE input is strictly forbidden from invoking the ESA-trained risk model due to lack of verified object/time mapping.

---

## C. Contradictions Found

| Item # | Topic | Document 1 (Higher Priority) | Document 2 (Lower Priority) | Contradiction Analysis & Severity |
|---|---|---|---|---|
| **C.1** | **Documentation File Placement vs. Canonical Tree** | `rules.md` (P1) & Workspace structure: Files reside in `docs/` (`docs/ORVEXA_PRD.md`, `docs/ORVEXA_TECHNICAL_DESIGN.md`, `docs/ORVEXA_RESEARCH.md`, `docs/ORVEXA_VIBE_CODING_SPEC.md`, `docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md`) except `rules.md` and `ORVEXA_TESTING.md` in root. | `docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md` (P5, lines 24–30): Lists all specification files directly in root `orvexa/` (`ORVEXA_PRD.md`, `ORVEXA_TECHNICAL_DESIGN.md`, etc.) and lists `AGENTS.md` instead of `rules.md`. | **Medium Severity**: Structural discrepancy between the canonical tree definition and the actual filesystem layout. `rules.md` takes precedence. |
| **C.2** | **Rulebook Filename Reference** | `rules.md` (P1, line 1): Self-identifies as `rules.md`. | `docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md` (P5, lines 9, 24, 205): Refers to `AGENTS.md` as the rulebook. | **Low Severity**: Naming mismatch (`rules.md` vs `AGENTS.md`). |
| **C.3** | **Experiment A9 in Config Directory** | `docs/ORVEXA_VIBE_CODING_SPEC.md` (P2, line 395) & `rules.md` (P1, line 147): Defines A9 as a mandatory experiment ("Single snapshot versus temporal summaries versus TCN"). | `docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md` (P5, lines 42–54): Lists experiment YAML files `A1` to `A8` and `A10` to `A14` in `configs/experiments/`, completely omitting `A9`. | **Medium Severity**: A9 is defined as an experiment ID, but has no config file in the canonical structure. A9 is an analytical multi-model comparison rather than a separate trained model. |
| **C.4** | **Allowed Framework Extensions (Scope)** | `rules.md` (P1, line 35): "Do not add React, FastAPI, Docker, a database, a Transformer, a GNN, or reinforcement learning unless a written requirement makes it necessary." | `docs/ORVEXA_PRD.md` (P4, lines 102–105) & `docs/ORVEXA_TECHNICAL_DESIGN.md` (P3, lines 229–243): Defines nice-to-have specifications including FastAPI endpoints and React frontend. | **Low Severity / Resolved by Scope Hierarchy**: Priority 1 explicitly clarifies that Streamlit and local execution are mandatory for v1/MVP, and nice-to-have extensions are strictly deferred. |
| **C.5** | **Readout Pooling Mechanism** | `rules.md` (P1, line 143): "Use masked pooling or the final valid representation." | `docs/ORVEXA_VIBE_CODING_SPEC.md` (P2, line 358) & `docs/ORVEXA_TECHNICAL_DESIGN.md` (P3, line 221): Specifies `masked_mean_pooling` explicitly with the exact formula: `(hidden * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)`. | **Low Severity**: Vibe Spec and Technical Design provide the exact mathematical implementation of masked mean pooling. |

---

## D. Missing Information

1. **Exact ESA Raw CSV Column Names & Header Alignment:**
   - While `docs/ORVEXA_VIBE_CODING_SPEC.md` provides a feature-group whitelist (e.g., `t_sigma_r`, `c_sigma_r`, `miss_distance`, `relative_speed`), actual raw headers in the ESA Zenodo archive `Collision Avoidance Challenge - Dataset.zip` contain exact column names (such as specific covariance matrix components, object identifiers, and solar indices) that must be dynamically validated.
   - *Status in Spec:* The specification intentionally requires running `python -m orvexa.cli audit-data` first to generate `reports/schema_report.json` and confirm the alias map rather than guessing.

2. **Event Ordering Key Availability in Raw Data:**
   - The specifications note that if an explicit CDM generation timestamp or epoch exists, it must be used for chronological ordering; if absent, the first-appearance row order serves as the fallback proxy.
   - The exact presence and format of timestamp columns (e.g., `c_j_date`, `epoch`) in `train_data.csv` must be confirmed during the data audit step.

3. **Missingness Indicator Strategy for TCN Sequences:**
   - `rules.md` and `docs/ORVEXA_VIBE_CODING_SPEC.md` state that static models receive median imputation + missingness indicators, while TCN receives imputed values + validity mask. It is underspecified whether per-feature missingness channels are concatenated to the TCN feature dimension or if the single temporal sequence mask `(batch, time)` is sufficient.

---

## E. Potential Data Leakage Issues

The documentation thoroughly addresses leakage risks. Below is the risk analysis and verification points:

| Leakage Vector | Specific Risk | Mitigation in Documentation | Validation Rule |
|---|---|---|---|
| **Target Leakage** | Final CDM `risk` or future CDM `risk` appearing in input features or temporal summaries. | Whitelist policy; `risk` is stripped from all feature tables; final target is joined only after prefix construction. | Assert `risk` not in feature columns; test target mutation does not change feature tensors. |
| **Temporal Horizon Leakage** | Using CDMs occurring closer to TCA than the cutoff horizon ($t < H$). | Filtering invariant: $\text{time\_to\_tca} \ge H$ for all rows in prefix. Final row ($t < H$) is never in input. | Unit test `test_no_post_cutoff_rows` across all horizons. |
| **Partition Leakage (Event Cross-Contamination)** | CDMs from the same conjunction event split across train and validation/test. | Split is performed strictly on unique `event_id` keys before building prefixes. | Unit test `test_event_partition_disjoint` asserts disjoint sets. |
| **Preprocessing & Transformation Leakage** | Fitting imputers, scalers, one-hot encoders, or outlier transforms on validation/test data. | Pipeline fits strictly on training split; validation/test are transformed with frozen artifacts. | Unit test `test_train_only_preprocessing` confirms frozen parameters. |
| **Calibration Leakage** | Fitting Platt or isotonic calibrators on test data or train data. | Calibrators are fitted strictly on validation predictions and evaluated once on test. | Calibrator fit receives only validation split predictions. |
| **Threshold Selection Leakage** | Selecting log-risk classification thresholds based on test set PR curves. | Thresholds are predetermined ($[-6.0, -5.0, -4.0]$) for sensitivity analysis. | Primary report outputs all three thresholds without optimization. |
| **Statistical Resampling Leakage** | Bootstrapping individual CDM rows as independent observations. | Resampling is strictly at the event level ($N=2000$ iterations). | Bootstrap unit is `event_id`. |
| **Cross-Dataset Contamination** | Merging Space-Track / CelesTrak TLE records into anonymized ESA CDMs. | Hard separation: TLE data is used solely for SGP4 orbital propagation demo. | TLE inputs cannot trigger ML risk prediction. |

---

## F. Recommended Corrections

1. **Synchronize Canonical Project Structure with Actual Directory Layout:**
   - Update `docs/ORVEXA_CANONICAL_PROJECT_STRUCTURE.md` to reflect that the specifications reside in `docs/` and that `rules.md` is the primary rulebook in root.

2. **Clarify Nature of Experiment A9:**
   - Formally document in `configs/` and documentation that Experiment `A9` is a composite synthesis/evaluation comparison (Snapshot XGBoost vs Temporal XGBoost vs TCN) that operates over the artifacts produced by `A3-A8`, and does not require a standalone training script.

3. **Standardize TCN Input Feature Dimension:**
   - Explicitly define the input tensor features for TCN as the normalized tabular feature set (with median imputation) plus the sequence validity mask `(batch, time)`.

4. **Retain Strict Whitelist for Schema Audit:**
   - Ensure `src/orvexa/schema.py` and `audit-data` generate `reports/schema_report.json` and raise descriptive errors if any mandatory whitelist column cannot be resolved.

---

## G. Questions Requiring Human Decision

1. **Specification File Location:**
   - Should specification files remain in `docs/` (as currently structured) or be moved to project root as depicted in `ORVEXA_CANONICAL_PROJECT_STRUCTURE.md`?  
   *(Recommendation: Keep in `docs/` with `rules.md` and `ORVEXA_TESTING.md` at root to maintain a clean repository root).*

2. **Experiment A9 Config Artifact:**
   - Should a `configs/experiments/A9_comparison.yaml` file be added to the canonical list, or should A9 remain designated as a multi-artifact evaluation report?  
   *(Recommendation: Treat A9 as an evaluation/reporting pipeline step comparing A3–A8 artifacts).*

3. **Auxiliary TLE Dataset Download Scope:**
   - Should automated downloading of Space-Track TLE history and CelesTrak active ephemeris be included in initial setup scripts, or purely deferred until the core ESA ML pipeline (Milestones 1–10) is verified?  
   *(Recommendation: Defer external TLE retrieval until core ESA experiments A1–A8 and test suites pass).*

---

## H. Final Implementation Readiness Assessment

### Compliance Check with Outlawed / Rejected Concepts:
- [x] **Computer Vision:** No computer vision references or requirements.
- [x] **Reinforcement Learning:** Excluded; non-goal confirmed.
- [x] **Debris Reuse / Circular Economy:** Excluded; non-goal confirmed (Zenodo 4986292 explicitly excluded).
- [x] **Autonomous Debris Capture / Maneuver Decisions:** Excluded; explicit non-goal and disclaimers defined.
- [x] **GNN as Required Model:** Excluded; primary deep model is strictly masked causal TCN.

### Readiness Rating: **READY FOR INITIAL SCAFFOLDING & DATA AUDIT**

**Summary:**  
The ORVEXA project specifications are exceptionally thorough, mathematically sound, and rigorously safeguarded against data leakage. The documents provide clear, non-negotiable boundaries, explicit architectural defaults, deterministic metrics, and comprehensive test invariants.

No code or data modifications have been made during this audit. Implementation can safely proceed to Milestone 1 (environment scaffolding, base configuration, data audit, and schema validation) once user decisions on structural alignment are confirmed.
