# Dataset & Provenance Card (`OMNI-NMDB-ERA5-OLR`)

## 1. Provenance & Alignment Protocol
* **Temporal Anchor:** All series are normalized to **UTC daily midpoints (`12:00:00 UTC`)** and **27-day Bartels solar rotation means** spanning `1979-01-01` to `2025-12-31`.
* **Quality Control Masks:**
  * **OMNIWeb:** Masked fill values (`999.9`, `9999.0`) and spacecraft cross-calibration transition epochs (IMP-8 to Wind/ACE transition in 1995–1998).
  * **NMDB Neutron Monitors:** Used barometric pressure- and efficiency-corrected count rates (`corr_for_efficiency`) for Oulu ($R_c = 0.8\text{ GV}$), Moscow ($R_c = 2.4\text{ GV}$), and Tokyo/Mt. Norikura ($R_c = 11.4\text{ GV}$). Flagged Ground Level Enhancements (GLEs) separately from Forbush Decreases so solar energetic proton spikes do not corrupt galactic cosmic-ray modulation inversions.
  * **NOAA CDR OLR & ERA5:** Applied land-sea and zenith-angle quality flags; regridded ERA5 ($0.25^\circ$) to match NOAA OLR ($1.0^\circ$) via conservative spherical remapping in `xarray`.

## 2. Temporal Leakage Audit
* **Lag Purge Buffer:** A **54-day ($2\times$ Bartels rotation) purge window** is enforced around every held-out Solar Cycle boundary and Forbush Decrease event window (`src/models/blocked_cv.py`).
* **Scaler Isolation:** All feature standardization (`RobustScaler`) and Kalman state-space parameters are fit strictly inside each training fold and applied forward-only to test folds.

## 3. Cautious Interpretation & Confounder Audit
* **Correlation vs. Microphysics:** While a statistically significant $\sim 27.2\text{-day}$ coherence ($p < 0.01$ vs. $AR(1)$ red noise) is observed between GCR flux and deep convective OLR ($\le 200\text{ W/m}^2$) over the Western Pacific Warm Pool during solar maxima, this pipeline treats the relationship strictly as an **empirical diagnostic**.
* **Confounding Pathways:** Co-variability at the 27-day scale includes contributions from Total/Spectral Solar Irradiance (TSI/UV) heating of the upper stratosphere and internal intraseasonal oscillations (Madden–Julian Oscillation, 30–60 days) whose harmonics can alias into the 27-day band if not windowed carefully.