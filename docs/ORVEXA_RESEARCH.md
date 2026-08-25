# ORVEXA Research and Industry Context

## 1. Purpose

This document gives the coding assistant and project team the domain context needed to make scientifically appropriate product and UI decisions. It explains the operator problem, existing approaches, dataset boundaries, competitor/reference projects, project risks, and the defensible position of ORVEXA.

## 2. Domain problem

Spacecraft operators may receive repeated Conjunction Data Messages when a spacecraft and another tracked object are predicted to approach one another. The information evolves as observations and orbit estimates are updated. The practical challenge is not merely producing one score; it is deciding which events deserve limited analyst attention early enough for investigation.

This creates four user pain points:

1. **Alert volume:** Many conjunction records may compete for analyst attention.
2. **Early uncertainty:** An early estimate can change substantially before TCA.
3. **Trust:** A ranking without calibration or explanation is difficult to operationalize.
4. **Changing conditions:** A model trained on earlier data may degrade on later data or under missing observations.

ORVEXA addresses these pain points as a research prototype through temporal prefixes, alert-budget ranking, calibration, explanations, temporal evaluation, and robustness testing.

## 3. Existing benchmark and related work

The ESA Kelvins Collision Avoidance Challenge established a public machine-learning benchmark in which participants used CDM information available up to two days before TCA to predict final collision risk.[1] The challenge attracted 96 teams and 862 submissions.[2]

Research on the benchmark has already included conventional and recurrent models such as Random Forest, MLP, RNN, LSTM, GRU, and sequence-to-sequence methods.[3] The Kessler open-source project also provides spacecraft collision-avoidance machine-learning and simulation tooling.[4]

Recent work makes the novelty boundary especially important. A 2026 study reports risk classification experiments involving class imbalance methods and threshold optimization.[5] A 2026 Acta Astronautica study formulates ESA-CDM conjunction prediction as full-horizon, uncertainty-aware sequence-to-sequence forecasting and evaluates Monte Carlo and TimeQuery Transformer models with quantile heads and Split Conformal Prediction.[6]

Therefore, ORVEXA should not claim to be the first collision-risk predictor, first sequence model, first multi-horizon model, or first uncertainty-aware model.

## 4. ORVEXA position

ORVEXA is positioned as a focused, reproducible, AI and Data Science study of **decision-support value**.

> ORVEXA asks whether a masked causal TCN adds useful value beyond an ESA-derived estimate and strong tabular baselines when events are evaluated with chronological leakage control, fixed alert budgets, probability calibration, explanation stability, and degraded-observation tests.

This is narrower and more defensible than claiming a new operational conjunction-assessment system.

## 5. Reference and competitor landscape

| Reference/project | What it provides | ORVEXA relationship |
|---|---|---|
| ESA Kelvins challenge | Public CDM benchmark and final-risk prediction task | Primary dataset and historical benchmark context. |
| Prior benchmark studies | Classical and recurrent ML comparisons | Baseline and related-work context; ORVEXA adds a controlled TCN/decision-support evaluation. |
| Kessler | Spacecraft collision-avoidance ML and simulation library | Domain reference; check GPL-3.0 before copying code. |
| OrbVeil | CDM parsing, TLE/SGP4 screening, collision-probability tooling | Optional supporting parser/screening module, not the core AI contribution. |
| Operational collision-avoidance services | High-fidelity data and procedures | ORVEXA is not a replacement; it lacks operational ephemerides, certification, and maneuver authority. |
| Recent uncertainty-aware Transformer work | Full-horizon sequence prediction, quantiles, conformal coverage | Important related work; ORVEXA differentiates through focused TCN comparison, alert-budget evaluation, drift, and robustness. |

## 6. Dataset context

### Primary ESA data

The official ESA dataset contains event sequences of CDMs and a final event-level `risk` target. The validated training archive contains approximately 162,634 rows, 103 columns, and 13,154 events. The data contain meaningful missingness, including approximately 32.5% missingness in `c_rcs_estimate` in the inspected training file.

The target is a self-computed ESA base-10 log-risk score. It should not be presented as a directly measured physical collision probability. The official competition test set is not an unbiased random sample and has special temporal restrictions; therefore, ORVEXA’s main scientific result uses a new event-level chronological split from labelled training data.[7]

### Space-Track TLE History

The Space-Track TLE History dataset contains historical orbital-element records indexed by NORAD ID and epoch. It is suitable for TLE parsing, SGP4 propagation, orbital-state plots, and an independent physics demonstration. It is not a labelled conjunction-risk dataset and is not row-wise joined to ESA data by default.[8]

### CelesTrak

CelesTrak supplies current catalogue GP/TLE/OMM data and is useful for a current-data demonstration. It is dynamic, so every retrieval must record endpoint, timestamp, and response hash. It does not supply the ESA final-risk label.[9]

## 7. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Final-CDM leakage | Inflated results and invalid thesis | Build prefixes before target join; test every horizon. |
| Event leakage | Repeated event information crosses partitions | Split unique events, then expand rows/sequences. |
| Horizon coverage loss | 5/7-day samples become small or biased | Report usable coverage; make 7-day conditional. |
| Extreme imbalance | Accuracy becomes misleading | PR-AUC, Recall@K, threshold sensitivity, event bootstrap. |
| MRE dominance | AI contribution becomes unclear | Mandatory with/without `max_risk_estimate` ablations. |
| TCN overfitting | Poor generalization | Small model, dropout, early stopping, chronological validation, seed repetition. |
| TCN does not win | Temptation to manipulate results | Report honestly; temporal-summary XGBoost may be the practical winner. |
| Calibration drift | High-risk probabilities become unreliable | Fit on validation, evaluate later windows, report drift. |
| Attribution overclaim | Correlation interpreted as causality | Label explanations as model behaviour only. |
| TLE misuse | Misleading live risk score | Keep TLE/SGP4 separate and disable arbitrary TLE risk scoring. |
| Scope explosion | Core experiments remain unfinished | Preserve mandatory hierarchy; make frontend and optional models last. |
| Dataset changes | Reproduction becomes difficult | Store source URLs, checksums, retrieval dates, and schema reports. |
| License conflict | Unintended redistribution obligations | Record third-party licenses and avoid copying GPL code casually. |

## 8. Research ethics and safety language

The dashboard and thesis must state that ORVEXA produces research estimates and is not an operational collision-avoidance authority. Do not display a score for an arbitrary live object as though it were an authoritative probability. Do not recommend spacecraft maneuvers. Do not hide model uncertainty, missing data, or negative findings.

## 9. Recommended UI/UX principles

The interface should prioritize comparison and transparency over decoration. The user should always see the selected horizon, model, split, event count, usable coverage, alert budget, and whether a probability is calibrated.

Ranked alerts should include enough context to make the ranking interpretable: event ID, target/chaser label when available, time to TCA, miss distance, predicted log-risk, calibrated high-risk probability, priority rank, and model version.

Reliability and robustness should be first-class pages rather than hidden in a report. A model that ranks well but is poorly calibrated or highly sensitive to missing observations should be visibly labelled as such.

## 10. What ORVEXA can legitimately claim

ORVEXA can claim that it:

- Implements a reproducible event-level ESA CDM pipeline.
- Compares an ESA-derived reference with static, temporal-summary, and TCN AI models.
- Studies the effect of warning horizon on prediction, coverage, and ranking.
- Evaluates alert-budget utility with Recall@K and related ranking metrics.
- Measures calibration and temporal calibration drift.
- Tests explanation stability and degraded-observation robustness.
- Provides a research dashboard and an independent TLE/SGP4 demonstration.

ORVEXA should not claim that it:

- Replaces ESA, Space-Track, 18th Space Defense Squadron, or operator procedures.
- Predicts a directly observed physical collision probability.
- Produces maneuver decisions.
- Is the first AI collision predictor or first temporal model in the domain.
- Combines TLE and ESA data into a labelled model without a demonstrated mapping.

## 11. References

[1]: https://kelvins.esa.int/collision-avoidance-challenge/data/ "ESA Kelvins Collision Avoidance Challenge data description"

[2]: https://arxiv.org/html/2008.03069v2 "Uriot et al., Spacecraft Collision Avoidance Challenge: design and results of a machine-learning competition"

[3]: https://www.sciencedirect.com/science/article/abs/pii/S246889672100094X "Predicting risk of satellite collisions using machine learning"

[4]: https://github.com/kesslerlib/kessler "Kessler collision-avoidance machine-learning library"

[5]: https://www.epj-conferences.org/articles/epjconf/abs/2026/10/epjconf_gcmm2025_02006/epjconf_gcmm2025_02006.html "Machine-learning-based risk classification of space-debris conjunction events"

[6]: https://www.sciencedirect.com/science/article/abs/pii/S0094576526005357 "Deep learning for full-horizon uncertainty-aware prediction of CDM sequences in spacecraft collision avoidance"

[7]: https://zenodo.org/records/4463683 "Official ESA Collision Avoidance Challenge dataset"

[8]: https://huggingface.co/datasets/juliensimon/space-track-tle-history "Space-Track TLE History dataset"

[9]: https://celestrak.org/NORAD/elements/ "CelesTrak NORAD GP element sets"

## 12. Final guidance

The best project is not the one with the largest number of models. It is the one that makes a precise claim, tests it without leakage, reports uncertainty and failure, and explains why its results matter to a constrained operator workflow.
