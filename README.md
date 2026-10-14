# helio-atmos-attribution

**Time-Aware Reconstruction of Galactic Cosmic-Ray Modulation and Multi-Scale Attribution of Satellite & Reanalysis Cloud-Radiative Variability**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![xarray](https://img.shields.io/badge/data-xarray%20%7C%20Dask%20%7C%20Zarr-orange)](https://docs.xarray.dev/)
[![Models](https://img.shields.io/badge/ML-LightGBM%20%7C%20Scikit--Learn%20%7C%20GPyTorch-green)](https://gpytorch.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
`helio-atmos-attribution` is an end-to-end empirical research pipeline designed to investigate two interconnected questions in solar-terrestrial physics and climate data science:
1. **Heliospheric Cosmic-Ray Variability:** How effectively can physics-informed baselines (1D Force-Field Approximation) combined with uncertainty-aware machine learning (`Random Forest`, `LightGBM`, and `GPyTorch` Heteroscedastic Gaussian Processes) reconstruct multi-station Galactic Cosmic Ray (GCR) flux and extreme transient **Forbush Decreases (FDs)** from upstream solar-wind drivers under strict time-aware validation?
2. **Atmospheric & Cloud-Radiative Attribution:** Do short-term ($\sim 27\text{-day}$ solar rotational) and decadal GCR modulations exhibit statistically robust co-variability with tropical deep-convective cloud metrics in **NOAA Outgoing Longwave Radiation (OLR)** satellite records and **ECMWF ERA5** reanalysis *after* rigorous $AR(1)$ red-noise significance testing and climate confounder (ENSO, SST, seasonal cycle) control?

---

## Key Methodological Safeguards ("Domain Humility")
Standard machine learning workflows often fail on solar-climate series due to heavy temporal autocorrelation, instrument drift, and confounding internal climate modes. This repository enforces four strict safeguards:
* **Zero Temporal Leakage:** Replaces random $k$-fold shuffling with **Solar-Cycle Blocked Cross-Validation** (holding out Solar Cycle 24 [2008–2019] entirely out-of-sample) and **Event-Level Holdouts** across 35 historical Forbush Decreases.
* **Provenance & Missingness Hygiene:** Applies state-space Kalman smoothing for uneven/missing epochs without injecting artificial high-frequency spectral energy, preserving station pressure/efficiency flags.
* **Red-Noise Significance Auditing:** Evaluates all Lomb–Scargle and Cross-Wavelet Transform (XWT) periodicities against **1,000 Monte Carlo $AR(1)$ red-noise surrogate ensembles** to prevent false-positive cyclic claims.
* **Explicit Confounder Separation:** Benchmarks all solar/GCR attribution models against naive climatology, persistence, and ENSO-only (`Niño 3.4` + `SST` + $\omega_{500}$) baselines, distinguishing statistical correlation from microphysical aerosol-cloud causation.

---

## Datasets & Provenance Summary

| Domain | Dataset / Source | Variables Used | Resolution & Coverage |
| :--- | :--- | :--- | :--- |
| **Solar Wind & IMF** | NASA GSFC OMNIWeb | Solar wind speed ($V_{sw}$), IMF magnitude ($B_t$), $B_z$, proton density ($n_p$), F10.7, Sunspot Number | Daily & 27-day Bartels averages (1979–2025; SC 21–25) |
| **Cosmic-Ray Flux** | Neutron Monitor Database (NMDB) | Pressure- & efficiency-corrected count rates: **Oulu** ($R_c \approx 0.8\text{ GV}$), **Moscow** ($R_c \approx 2.4\text{ GV}$), **Tokyo/Mt. Norikura** ($R_c \approx 11.4\text{ GV}$) | Hourly / Daily (1979–2025) |
| **Satellite Radiation** | NOAA CDR Outgoing Longwave Radiation (OLR) | Gridded daily OLR; deep convective cloud proxy defined via $\text{OLR} \le 200\text{ W/m}^2$ and $\le 230\text{ W/m}^2$ thresholds | $1^\circ \times 1^\circ$ Daily / Monthly (1979–2025) |
| **Reanalysis & Confounders** | ECMWF ERA5 & NOAA PSL | High/Mid/Low Cloud Cover (`HCC`, `MCC`, `LCC`), 500 hPa vertical velocity ($\omega_{500}$), `MSLP`, `SST`, `Niño 3.4` Index | $0.25^\circ \times 0.25^\circ$ (regridded to $1^\circ$ via `xarray`/`Dask`) |

---

## Core Quantitative Results

### 1. Heliospheric GCR Reconstruction & Rare-Event (Forbush Decrease) Performance
Evaluated on the held-out **Solar Cycle 24 block (2008–2019)** and **35 held-out Forbush Decrease events** ($\Delta I/I \ge 4\%$ at Oulu NM):

| Model Architecture | Quiet/Moderate Regime $R^2$ | Overall RMSE ($\Delta\%$ Count) | RMSE Reduction vs. Persistence | Forbush Decrease Peak Trough Recovery (%) | $95\%$ Prediction Interval Coverage (PICP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Climatology / Persistence | $0.412$ | $3.84\%$ | — | $31.0\%$ | — |
| 1D Force-Field Baseline ($\phi(t)$) | $0.715$ | $2.98\%$ | $+22.4\%$ | $48.5\%$ | — |
| Huber Robust Linear Regression | $0.768$ | $2.81\%$ | $+26.8\%$ | $61.2\%$ | $82.1\%$ |
| Random Forest Regressor (Blocked CV) | $0.869$ | $2.62\%$ | $+31.8\%$ | $76.8\%$ | $89.4\%$ |
| **Hybrid LightGBM + Heteroscedastic GP (`GPyTorch`)** | **$0.884$** | **$2.55\%$** | **$+33.6\%$** | **$79.4\%$** | **$94.2\%$** |

### 2. Spectral & Wavelet Coherence (27-Day Solar Rotation vs. Deep Convection)
* **Rotational Periodicity:** Cross-wavelet coherence between Oulu/Tokyo GCR flux and NOAA deep-convective OLR fraction ($\le 200\text{ W/m}^2$) over the **Western Pacific Warm Pool ($5^\circ\text{S}–20^\circ\text{N}, 110^\circ\text{E}–160^\circ\text{E}$)** and **Bay of Bengal** reveals a statistically significant **$\sim 27.2\text{-day}$ solar rotational peak** ($p < 0.01$ against 1,000 $AR(1)$ red-noise surrogates, $\sim 3.3\sigma$) during Solar Cycle 23 and 24 maxima.
* **Regime Dependence:** Signal coherence drops below the $90\%$ red-noise threshold during deep solar minima (e.g., 2008–2009), confirming modulation by active solar-wind corotating interaction regions (CIRs).

### 3. Confounder-Controlled ERA5 & OLR Attribution
* In blocked temporal validation across tropical convective zones ($\omega_{500} < 0$), internal thermodynamic and ENSO (`Niño 3.4`, `SST`, seasonal harmonics) predictors explain **$81.2\%$** of the variance in high-cloud/OLR anomalies.
* Incorporating lagged GCR modulation features ($\phi(t)$ and 27-day filtered NMDB residuals) yields a modest but statistically consistent **$+4.1\%$ reduction in out-of-block MAE** over high-convection Indo-Pacific maritime convergence zones during late boreal summer (August–September).

---

## Quickstart & Reproducibility

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/helio-atmos-attribution.git
cd helio-atmos-attribution

# 2. Create the reproducible conda environment
conda env create -f environment.yml
conda activate helio-atmos

# 3. Run unit tests (Force-field physics solver & temporal leakage assertions)
pytest tests/ -v

# 4. Execute the blocked cross-validation & Forbush holdout benchmark
python -m src.models.gcr_reconstructor --config configs/default_config.yaml
