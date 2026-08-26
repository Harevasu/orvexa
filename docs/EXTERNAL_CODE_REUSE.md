# ORVEXA External Code Reuse Record

This document provides a formal, comprehensive ledger of all external software components, mathematical formulations, and algorithmic architectures evaluated, adapted, or reimplemented within the ORVEXA codebase from the reference repositories located under `reference_repo/`.

---

## Code Reuse Ledger

| Reference Repository | Source File & Function / Class | Upstream License | Permitted Reuse? | Reuse Type | ORVEXA Destination File | Modifications & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`orbveil`** | `src/orbveil/core/probability.py`<br>`_project_to_bplane`, `compute_pc_foster` | Apache-2.0 | **YES** | Snippet & Algorithmic Adaptation | [`src/orvexa/models_physics.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/models_physics.py) | **Modifications**: Reimplemented B-plane projection and Foster (1992) 2D Gaussian quadrature in pure Python without forcing `scipy` runtime dependency. Retained mathematical conventions while adding robust covariance determinant singularity checks.<br>**Rationale**: Foster 2D integration is the spaceflight industry standard for conjunction risk; adopting the verified polar coordinate integration bounds ensures scientific fidelity. |
| **`CARA_Analysis_Tools`** | `DistributedMatlab/Maximum2DPc/FrisbeeMaxPc.m`<br>`FrisbeeMaxPc` | NASA Open Source Agreement (NOSA v1.3) | **YES** | Algorithmic Reimplementation | [`src/orvexa/models_physics.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/models_physics.py) | **Modifications**: Ported Frisbee (2015) maximum collision probability under covariance dilution from MATLAB to vectorizable Python functions.<br>**Rationale**: Frisbee maximum $P_c$ serves as an analytical upper-bound benchmark for covariance dilution against raw ESA `max_risk_estimate`. |
| **`CARA_Analysis_Tools`** | `DistributedMatlab/Utils/CoordinateTransformations/`<br>ECI, RTN/RIC transforms | NASA Open Source Agreement (NOSA v1.3) | **YES** | Algorithmic Reimplementation | [`src/orvexa/orbit/coordinate_utils.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/orbit/coordinate_utils.py) | **Modifications**: Re-expressed 3D rotational matrices ($R$, $T$, $N$) in pure Python with singular-velocity edge case fallbacks.<br>**Rationale**: Prevents frame mismatch errors between satellite-centric RTN and inertial J2000 frames. |
| **`pytorch-tcn`** | `pytorch_tcn/tcn.py`, `conv.py`<br>`TemporalConv1d`, `TemporalResidualBlock` | MIT License | **YES** | Architectural Translation | [`src/orvexa/models_tcn.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/models_tcn.py) | **Modifications**: Rebuilt causal 1D dilated residual blocks to strictly accept ORVEXA validity masks (`[batch, time]`), exact $(k-1) \cdot d$ left-padding, and final-valid-step representation pooling. Implemented pure NumPy forward engine fallback for offline validation.<br>**Rationale**: Bai et al. (2018) causal residual convolutions have proven superior to autoregressive RNNs for long sequences while preventing future lookahead. |
| **`python-sgp4`** | `sgp4/io.py`<br>`compute_checksum` | MIT License | **YES** | Snippet & Function Adaptation | [`src/orvexa/orbit/tle_parser.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/orbit/tle_parser.py)<br>[`src/orvexa/orbit/validation.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/orbit/validation.py) | **Modifications**: Integrated standard NORAD modulo-10 checksum calculation and full 69-character format parsing with LEO apogee/perigee altitude derivation.<br>**Rationale**: Checksum and field parsing are strict standard protocols; reusing Brandon Rhodes' verified parsing logic eliminates subtle off-by-one errors. |
| **`satkit`** | `src/frametransform/mod.rs`<br>TEME to ECEF / Geodetic | MIT License | **YES** | Architectural Reimplementation | [`src/orvexa/orbit/coordinate_utils.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/orbit/coordinate_utils.py) | **Modifications**: Translated Rust Bowring's geodetic latitude/longitude/altitude algorithm on WGS-84 into Python with GMST sidereal rotation.<br>**Rationale**: Bowring's method is the exact closed-form standard for satellite ground track computation. |
| **`kessler`** | `src/data/`<br>Autoregressive LSTM & dataset loading | GNU General Public License v3.0 (GPL-3.0) | **NO (Copying Prohibited)** | Reference Only / Conceptual Cleanroom | None (No code copied) | **Cleanroom Action**: Source code was NOT copied or imported due to GPL-3.0 viral copyleft restrictions. ORVEXA's temporal pipeline uses non-autoregressive causal TCNs and tabular gradient boosting instead. |
| **`SETTLE`** | Entire Repository | No License Specified (All Rights Reserved) | **NO (Copying Prohibited)** | Reference Only | None (No code copied) | **Action**: No source code was copied or referenced due to lack of an open-source grant. |
| **`OrbitalShield`** | Entire Repository | BSD 3-Clause | **YES (Reference Only)** | Reference Only | None | **Action**: Used for architecture comparison only; no Ada source code reused. |

---

## Detailed Attribution Notices

### 1. Foster (1992) 2D Collision Probability
- **Original Authors**: J. L. Foster, H. S. Estes (NASA Lyndon B. Johnson Space Center, 1992).
- **Reference Implementation**: `reference_repo/orbveil` (`src/orbveil/core/probability.py`), Copyright (c) 2024 orbveil contributors, Apache License 2.0.
- **ORVEXA Implementation**: [`src/orvexa/models_physics.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/models_physics.py#L104-L180).

### 2. Frisbee Maximum Collision Probability
- **Original Authors**: J. H. Frisbee (2015), S. Alfano (2005).
- **Reference Implementation**: `reference_repo/CARA_Analysis_Tools` (`DistributedMatlab/Maximum2DPc/FrisbeeMaxPc.m`), NASA Conjunction Assessment Risk Analysis (CARA), NOSA v1.3.
- **ORVEXA Implementation**: [`src/orvexa/models_physics.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/models_physics.py#L183-L215).

### 3. Causal Temporal Convolutional Networks
- **Original Authors**: S. Bai, J. Z. Kolter, V. Koltun (2018), "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling".
- **Reference Implementation**: `reference_repo/pytorch-tcn`, Copyright (c) 2021 Paul Krug, MIT License.
- **ORVEXA Implementation**: [`src/orvexa/models_tcn.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/models_tcn.py#L65-L200).

### 4. SGP4 Orbit Propagation & TLE Checksums
- **Original Authors**: F. R. Hoots, R. L. Roehrich (1980), D. Vallado et al. (2006), Brandon Rhodes (python-sgp4).
- **Reference Implementation**: `reference_repo/python-sgp4`, Copyright (c) 2012-2024 Brandon Rhodes, MIT License.
- **ORVEXA Implementation**: [`src/orvexa/orbit/tle_parser.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/orbit/tle_parser.py), [`src/orvexa/orbit/propagation.py`](file:///c:/Users/Zyren/Documents/orvexa/src/orvexa/orbit/propagation.py).
