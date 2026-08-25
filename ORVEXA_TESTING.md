# ORVEXA Testing Strategy

## 1. Testing objective

Testing must prove both software correctness and scientific validity. A pipeline that trains successfully but leaks the final CDM, mixes events across partitions, or fits preprocessing globally is not correct.

The required testing layers are unit tests, data-contract tests, leakage tests, model smoke tests, evaluation tests, dashboard tests, and reproducibility checks.

## 2. Test commands

```bash
pytest -q
pytest tests/test_event_builder.py -q
pytest tests/test_split_leakage.py -q
pytest tests/test_preprocessing.py -q
pytest tests/test_metrics.py -q
python -m orvexa.cli audit-data --config configs/base.yaml
python -m orvexa.cli build-events --config configs/base.yaml
python -m orvexa.cli make-splits --config configs/base.yaml
```

The CI test suite must run without downloading the large ESA archive. Use a small checked-in fixture with synthetic structure but real column names and representative missingness. Production reports must use the real ESA data; fixtures are only for code tests.

## 3. Testing principles

Tests must be deterministic with seed 42 unless a test specifically checks seed variation. Tests must be small and fast. Public functions must validate inputs. Every bug fix must add a regression test. Do not weaken an assertion merely to make a failing test pass.

Tests must distinguish:

- **Data correctness:** values, types, ordering, and missingness.
- **Scientific correctness:** target and cutoff rules.
- **Model correctness:** tensor shapes, outputs, and saved artifacts.
- **Operational correctness:** ranking, calibration, dashboard states, and errors.

## 4. Required data tests

| Test | Required assertion |
|---|---|
| Required columns | `event_id`, `time_to_tca`, and `risk` must exist. |
| Numeric conversion | `risk` and `time_to_tca` must be numeric or fail clearly. |
| Target availability | Final event row has non-null `risk`. |
| Target identity | Event target equals final-row `risk`. |
| Duplicate audit | Exact duplicates and duplicate event/time pairs are reported. |
| Event grouping | All rows in an event have the same event key. |
| Ordering | Prefix order is oldest to newest; time-to-TCA decreases toward the anchor. |
| Horizon filter | Every input row satisfies `time_to_tca >= H`. |
| Anchor | Anchor is the smallest qualifying time-to-TCA. |
| Coverage | Events with no qualifying row are excluded and counted. |
| Sequence length | Length equals the number of qualifying rows retained. |
| Maximum length | No sequence exceeds configured T; truncation count is reported. |
| Feature registry | Unknown columns are excluded and logged. |
| Target exclusion | `risk` is absent from all feature lists. |
| ID exclusion | `event_id`, object identifiers, and mission identifiers are excluded by default. |

## 5. Leakage tests

### Test: final target is not an input

Build a prefix and assert that no feature vector contains the final event target. Test both static and sequence feature constructors.

### Test: post-cutoff rows are absent

For every horizon H and every generated sequence, assert:

```python
assert (prefix["time_to_tca"] >= H).all()
```

### Test: target join happens after prefix construction

Mutate the final-row risk in a fixture and confirm that the input vectors do not change while the target changes.

### Test: partition event disjointness

```python
assert set(train.event_id).isdisjoint(validation.event_id)
assert set(train.event_id).isdisjoint(test.event_id)
assert set(validation.event_id).isdisjoint(test.event_id)
```

### Test: shared preprocessing is not refit

Fit a transformer on training data, transform validation data, mutate validation values, and confirm the fitted statistics do not change.

### Test: no future sequence information

Change an excluded post-cutoff CDM and assert that the horizon-specific input tensor is unchanged.

## 6. Feature and preprocessing tests

Test median imputation, missingness indicators, categorical unknown handling, angle sine/cosine transforms, heavy-tail transforms, standardization, and feature-order persistence.

Assertions must include:

- Output has no unexpected NaN or infinity.
- Train-fitted transformers reproduce identical results after save/load.
- Validation-only categories do not crash one-hot encoding.
- Feature order before and after serialization is identical.
- Missingness indicators differ when values are removed.
- Scaling uses training statistics only.
- `max_risk_estimate` is present only in with-MRE variants.

## 7. TCN tests

Use a fixture with variable sequence lengths of 1, 3, and 23.

Required assertions:

- Input storage shape is `(batch, time, features)`.
- Model-entry shape is `(batch, features, time)`.
- Mask shape is `(batch, time)`.
- Valid positions have mask 1 and padding positions have mask 0.
- Left padding places the newest valid CDM at the final timestep.
- A change in padded values does not change masked output.
- A change in a future timestep outside the prefix does not change output.
- Regression output shape is `(batch,)` or `(batch, 1)` consistently.
- Classification output shape is consistent.
- Save/load reproduces predictions within a documented floating-point tolerance.
- A one-batch training smoke test completes without shape or NaN errors.

## 8. Model tests

For each model adapter:

- `fit()` accepts the documented data type.
- `predict_risk()` returns one prediction per event.
- Predictions are finite.
- Save/load preserves predictions.
- Missing artifacts produce clear errors.
- Feature-order mismatch stops evaluation.
- Classification probabilities are in `[0, 1]`.
- Random-state configuration is persisted.

## 9. Evaluation tests

Use a small deterministic fixture with known rankings and targets.

Required assertions:

- Higher/less-negative risk is ranked ahead of lower risk.
- Recall@K uses event counts, not row counts.
- Top-K count is deterministic for a fixed dataset and tie rule.
- NDCG handles all-negative and all-positive edge cases.
- PR-AUC handles a split with no positives by returning a documented unavailable status, not a misleading zero.
- Spearman handles constant targets with a documented null result.
- Brier score validates probability range.
- ECE reports bin counts.
- Bootstrap resamples event IDs, never CDM rows.
- Confidence interval bounds are ordered.

## 10. Calibration tests

Test Platt and isotonic fitting on validation predictions only. Verify that the final test labels are not passed into the calibrator’s fit method. Test small-positive-count behavior and produce a warning when calibration is statistically unstable.

## 11. Robustness tests

For each perturbation function:

- Fixed seed gives identical masked/noisy data.
- Masking never changes the target.
- Perturbation fraction is within tolerance of configured level.
- Reduced sequence retains only valid latest timesteps.
- Robustness results include clean and degraded metrics.
- The dashboard can load robustness artifacts even when one optional perturbation is unavailable.

## 12. TLE/SGP4 tests

Use a known valid fixture TLE/OMM record and test:

- Epoch parses correctly.
- NORAD ID is retained.
- Position and velocity units are labelled.
- Invalid TLE produces a useful validation error.
- SGP4 non-zero error code produces a warning or failure according to configuration.
- Propagation interval is explicit.
- TLE-only input cannot invoke ESA risk prediction.
- Source URL and retrieval time are saved.

## 13. CLI and artifact tests

Run each CLI command on a small fixture. Verify that expected files are created and that rerunning with the same seed produces equivalent metrics and hashes where deterministic.

Test failure cases:

- Missing raw file.
- Corrupt CSV.
- Missing required column.
- Empty event group.
- No horizon coverage.
- Model artifact absent.
- Metrics artifact malformed.
- Feature-order mismatch.
- Invalid experiment ID.
- Unsupported horizon.

## 14. Dashboard tests

Use Streamlit component tests or a lightweight smoke test for page loading. Verify:

- Missing artifacts show an actionable empty state.
- Empty horizon coverage does not crash.
- Alert budget filters use 1%, 5%, and 10%.
- Table columns are present.
- Event selection displays details.
- Explanation page displays a non-causal disclaimer.
- Reliability page displays Brier/ECE and bin counts.
- Robustness page displays clean/degraded comparison.
- TLE page never displays an ESA-risk score for TLE-only input.
- Every page contains the operational limitation statement.

## 15. Coverage expectations

Target minimums:

| Area | Minimum expectation |
|---|---:|
| Core data/event/split modules | 90% line coverage |
| Preprocessing and metrics | 90% line coverage |
| Model adapters | 80% line coverage |
| TCN shape/mask logic | 90% line coverage |
| Orbit adapter | 80% line coverage |
| Dashboard utility functions | 70% line coverage |
| Overall project | 80% line coverage where practical |

Coverage is not a substitute for scientific tests. A high coverage number with no leakage assertions is insufficient.

## 16. Reproducibility test

On a clean environment:

1. Install the lock file.
2. Run the data audit.
3. Build event views.
4. Create splits.
5. Train A8 at H=2.
6. Evaluate and generate artifacts.
7. Launch the dashboard.

The README must document expected commands, runtime, hardware assumptions, and artifact locations. The final repository must contain a reproducibility report.

## 17. Definition of done for code changes

A code change is complete only when:

- The affected requirement is identified.
- The implementation is type-consistent and documented.
- New or changed behavior has tests.
- Relevant tests pass.
- No leakage invariant is weakened.
- Configuration and artifact schemas are updated if needed.
- README or technical documentation is updated.
- The change is small enough to review.
