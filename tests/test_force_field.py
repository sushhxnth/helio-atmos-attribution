import numpy as np
import pandas as pd
from src.physics.force_field import (
    rigidity_to_kinetic_energy,
    modulated_gcr_spectrum,
    estimate_modulation_potential_from_nm,
    kalman_rts_smooth_1d,
)


def test_force_field_monotonic_in_phi():
    energies = np.linspace(0.5, 20.0, 50)
    j_quiet = modulated_gcr_spectrum(energies, phi_mv=350.0)
    j_active = modulated_gcr_spectrum(energies, phi_mv=950.0)
    assert np.all(j_quiet > j_active)


def test_modulation_inversion_impulse_recovery():
    idx = pd.date_range("2020-01-01", periods=10, freq="1D")
    counts = pd.Series([1.0, 1.0, 0.94, 0.92, 0.95, 0.98, 1.0, 1.0, 1.0, 1.0], index=idx)
    phi = estimate_modulation_potential_from_nm(counts, cutoff_rigidity_gv=0.8)
    assert phi.iloc[3] > phi.iloc[0]


def test_kalman_smoother_interpolates_missing_gaps():
    idx = pd.date_range("2020-01-01", periods=30, freq="1D")
    s = pd.Series(np.sin(np.linspace(0, 3.14, 30)), index=idx, name="signal")
    s.iloc[10:15] = np.nan
    smoothed, std = kalman_rts_smooth_1d(s)
    assert smoothed.isna().sum() == 0
    assert (std.iloc[10:15] > std.iloc[0:5].mean()).all()