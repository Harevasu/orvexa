# Third-Party Notices and Licenses

ORVEXA uses open-source libraries and public datasets. This document records all external dependencies, references, versions, and license notices in accordance with `rules.md`.

---

## 1. Datasets

### ESA Kelvins Collision Avoidance Challenge Dataset
- **Source:** European Space Agency (ESA) Kelvins / Zenodo (Record 4463683)
- **URL:** https://zenodo.org/records/4463683
- **License:** Open Access / ESA Challenge Terms
- **Usage:** Primary benchmark dataset for conjunction event risk prioritization. Used strictly in unmodified read-only form.

### Space-Track TLE History (Auxiliary)
- **Source:** Space-Track.org / Julien Simon HuggingFace Dataset
- **URL:** https://huggingface.co/datasets/juliensimon/space-track-tle-history
- **License:** Open Data / Space-Track User Agreement
- **Usage:** Supporting dataset for standalone SGP4 propagation and orbital visualization demonstration. Completely decoupled from ESA ML risk scoring.

### CelesTrak GP Data (Auxiliary)
- **Source:** CelesTrak
- **URL:** https://celestrak.org/NORAD/elements/
- **License:** Public Domain / Open Educational Data
- **Usage:** Real-time / recent TLE demonstration data.

---

## 2. Third-Party Software Libraries

| Library | License | URL | Purpose |
|---|---|---|---|
| **Python SGP4** | MIT | https://github.com/brandon-rhodes/python-sgp4 | Fast orbital propagation of two-line element sets. |
| **PyTorch-TCN** | MIT | https://github.com/paul-krug/pytorch-tcn | Reference Temporal Convolutional Network architecture. |
| **Kessler** | GPL-3.0 | https://github.com/kesslerlib/kessler | Domain reference for conjunction analysis equations. Not directly bundled. |
| **OrbVeil** | Apache-2.0 | https://github.com/ncdrone/orbveil | Optional parser / orbital screening utilities. |
| **PyTorch** | BSD-3-Clause | https://pytorch.org | Deep learning framework. |
| **scikit-learn** | BSD-3-Clause | https://scikit-learn.org | Classical machine learning and calibration tools. |
| **XGBoost** | Apache-2.0 | https://xgboost.readthedocs.io | Gradient boosted decision trees. |
| **SHAP** | MIT | https://github.com/shap/shap | Tree and model explainability. |
| **Streamlit** | Apache-2.0 | https://streamlit.io | Interactive research decision support dashboard. |
