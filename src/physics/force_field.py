from __future__ import annotations
import numpy as np
import pandas as pd

PROTON_REST_MASS_GEV = 0.938272


def rigidity_to_kinetic_energy(rigidity_gv: np.ndarray | float, z: int = 1, a: int = 1) -> np.ndarray | float:
    r_term = (z / a) * np.asarray(rigidity_gv)
    return np.sqrt(r_term**2 + PROTON_REST_MASS_GEV**2) - PROTON_REST_MASS_GEV


def local_interstellar_spectrum(energy_gev: np.ndarray) -> np.ndarray:
    p = np.sqrt(energy_gev * (energy_gev + 2.0 * PROTON_REST_MASS_GEV))
    beta = p / (energy_gev + PROTON_REST_MASS_GEV)
    return 1.9e4 * (p**-2.78) / (1.0 + 0.4866 * (p**-2.51)) * beta


def modulated_gcr_spectrum(energy_gev: np.ndarray, phi_mv: float, z: int = 1, a: int = 1) -> np.ndarray:
   
    phi_gev = (z / a) * (phi_mv / 1000.0)
    e0 = PROTON_REST_MASS_GEV
    e_shifted = energy_gev + phi_gev
    j_lis = local_interstellar_spectrum(e_shifted)
    phase_space = (energy_gev * (energy_gev + 2.0 * e0)) / (e_shifted * (e_shifted + 2.0 * e0))
    return j_lis * phase_space


def estimate_modulation_potential_from_nm(
    nm_count_norm: pd.Series,
    cutoff_rigidity_gv: float = 0.8,
    phi_ref_mv: float = 440.0,
    sensitivity_alpha: float = -0.00041,
) -> pd.Series:
   
    rigidity_scaling = 1.0 + 0.085 * (cutoff_rigidity_gv - 0.8)
    delta_rel = (nm_count_norm - 1.0) * rigidity_scaling
    phi_t = phi_ref_mv + (delta_rel / sensitivity_alpha)
    return phi_t.clip(lower=150.0, upper=1800.0).rename("phi_mv")


def kalman_rts_smooth_1d(
    series: pd.Series,
    process_var: float = 1e-3,
    obs_var: float = 1e-2,
) -> Tuple[pd.Series, pd.Series]:
    
    y = series.to_numpy(dtype=float)
    n = len(y)
    x_pred = np.zeros(n)
    p_pred = np.zeros(n)
    x_filt = np.zeros(n)
    p_filt = np.zeros(n)

    # Initialize with first valid observation
    first_valid = np.nanmean(y[:30]) if not np.isnan(np.nanmean(y[:30])) else 0.0
    x_curr = first_valid
    p_curr = 1.0

    for t in range(n):
        x_pred[t] = x_curr
        p_pred[t] = p_curr + process_var
        if not np.isnan(y[t]):
            k_gain = p_pred[t] / (p_pred[t] + obs_var)
            x_filt[t] = x_pred[t] + k_gain * (y[t] - x_pred[t])
            p_filt[t] = (1.0 - k_gain) * p_pred[t]
        else:
            x_filt[t] = x_pred[t]
            p_filt[t] = p_pred[t]
        x_curr, p_curr = x_filt[t], p_filt[t]

    # Backward RTS smoothing pass
    x_smooth = np.copy(x_filt)
    p_smooth = np.copy(p_filt)
    for t in range(n - 2, -1, -1):
        c_gain = p_filt[t] / p_pred[t + 1]
        x_smooth[t] = x_filt[t] + c_gain * (x_smooth[t + 1] - x_pred[t + 1])
        p_smooth[t] = p_filt[t] + (c_gain**2) * (p_smooth[t + 1] - p_pred[t + 1])

    return (
        pd.Series(x_smooth, index=series.index, name=f"{series.name}_kalman"),
        pd.Series(np.sqrt(np.maximum(p_smooth, 1e-9)), index=series.index, name=f"{series.name}_std"),
    )