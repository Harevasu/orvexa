# ORVEXA Data Directory

This directory manages datasets across the ORVEXA pipeline lifecycle.

## Critical Data Handling Invariants

1. **Immutable Raw Data:**
   - Files in `data/raw/esa/` (specifically `train_data.csv`) must NEVER be modified, overwritten, or edited in place.
   - Raw datasets are the ground truth for schema audits and reproducible pipeline execution.

2. **No Data Leakage:**
   - `data/processed/` contains derived event views, sequence tensors, and snapshots constructed strictly using rows available before the warning horizon cutoff ($t \ge H$).
   - The final event target risk is joined to the event record only after prefix filtering is complete.

3. **Subdirectory Structure:**
   - `raw/esa/`: Original ESA Challenge CSV dataset (`train_data.csv`).
   - `raw/space_track/`: Auxiliary Space-Track TLE history data (deferred).
   - `raw/celestrak/`: Auxiliary CelesTrak ephemeris data (deferred).
   - `interim/`: Intermediate extractions and temporary cache.
   - `processed/`: Horizon-filtered tables (`event_views/`, `sequences/`, `snapshots/`).
   - `manifests/`: Checksum manifests, schema reports, split manifests.
