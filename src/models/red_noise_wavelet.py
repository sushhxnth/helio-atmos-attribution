from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from scipy.signal import lombscargle, csd


def estimate_ar1_alpha(series: np.ndarray) -> float:
    """Estimate lag-1 autocorrelation coefficient alpha for an AR(1) red-noise process."""
    clean = series[~np.isnan(series)]
    clean = clean - np.mean(clean)
    if len(clean) < 3:
        return 0.0
    c0 = np.dot(clean, clean) / len(clean)
    c1 = np.dot(clean[:-1], clean[1:]) / (len(clean) - 1)
    return float(np.clip(c1 / c0, 0.0, 0.995))


def lomb_scargle_red_noise_test(
    time_days: np.ndarray,
    signal: np.ndarray,
    periods_days: np.ndarray,
    n_surrogates: int = 1000,
    confidence_level: float = 0.99,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Compute normalized Lomb-Scargle periodogram and evaluate empirical significance
    contours against `n_surrogates` Monte Carlo AR(1) red-noise realizations.
    """
    rng = np.random.default_rng(random_state)
    valid = ~np.isnan(signal)
    t = time_days[valid].astype(float)
    y = signal[valid].astype(float)
    y = (y - np.mean(y)) / (np.std(y) + 1e-9)

    angular_freqs = 2.0 * np.pi / periods_days
    obs_power = lombscargle(t, y, angular_freqs, normalize=True)

    alpha = estimate_ar1_alpha(y)
    sigma_eps = np.sqrt(max(1.0 - alpha**2, 1e-6))
    n_pts = len(t)

    surrogate_powers = np.zeros((n_surrogates, len(periods_days)))
    for i in range(n_surrogates):
        eps = rng.normal(0.0, sigma_eps, size=n_pts)
        red = np.zeros(n_pts)
        red[0] = eps[0]
        for k in range(1, n_pts):
            red[k] = alpha * red[k - 1] + eps[k]
        red = (red - np.mean(red)) / (np.std(red) + 1e-9)
        surrogate_powers[i, :] = lombscargle(t, red, angular_freqs, normalize=True)

    threshold_95 = np.quantile(surrogate_powers, 0.95, axis=0)
    threshold_99 = np.quantile(surrogate_powers, confidence_level, axis=0)
    p_values = np.mean(surrogate_powers >= obs_power[None, :], axis=0)

    return {
        "periods_days": periods_days,
        "observed_power": obs_power,
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
        "p_values": p_values,
        "ar1_alpha": np.array([alpha]),
    }


def cross_spectral_coherence_red_noise(
    series_x: pd.Series,
    series_y: pd.Series,
    nperseg: int = 256,
    n_surrogates: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute magnitude-squared coherence between GCR modulation and atmospheric OLR/cloud
    anomalies, evaluating p-values against paired AR(1) red-noise surrogates.
    """
    rng = np.random.default_rng(random_state)
    df_pair = pd.concat([series_x, series_y], axis=1).dropna()
    x = df_pair.iloc[:, 0].to_numpy()
    y = df_pair.iloc[:, 1].to_numpy()
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    freqs, pxy = csd(x, y, fs=1.0, nperseg=nperseg)
    _, pxx = csd(x, x, fs=1.0, nperseg=nperseg)
    _, pyy = csd(y, y, fs=1.0, nperseg=nperseg)
    obs_coh = (np.abs(pxy) ** 2) / (pxx.real * pyy.real + 1e-12)

    ax = estimate_ar1_alpha(x)
    ay = estimate_ar1_alpha(y)
    sx = np.sqrt(max(1.0 - ax**2, 1e-6))
    sy = np.sqrt(max(1.0 - ay**2, 1e-6))
    n = len(x)

    surr_coh = np.zeros((n_surrogates, len(freqs)))
    for i in range(n_surrogates):
        rx = np.zeros(n)
        ry = np.zeros(n)
        ex = rng.normal(0.0, sx, size=n)
        ey = rng.normal(0.0, sy, size=n)
        for k in range(1, n):
            rx[k] = ax * rx[k - 1] + ex[k]
            ry[k] = ay * ry[k - 1] + ey[k]
        _, s_pxy = csd(rx, ry, fs=1.0, nperseg=nperseg)
        _, s_pxx = csd(rx, rx, fs=1.0, nperseg=nperseg)
        _, s_pyy = csd(ry, ry, fs=1.0, nperseg=nperseg)
        surr_coh[i, :] = (np.abs(s_pxy) ** 2) / (s_pxx.real * s_pyy.real + 1e-12)

    valid_f = freqs > 0
    periods = 1.0 / freqs[valid_f]
    return pd.DataFrame({
        "period_days": periods,
        "coherence": obs_coh[valid_f],
        "red_noise_95": np.quantile(surr_coh[:, valid_f], 0.95, axis=0),
        "red_noise_99": np.quantile(surr_coh[:, valid_f], 0.99, axis=0),
        "p_value": np.mean(surr_coh[:, valid_f] >= obs_coh[None, valid_f], axis=0),
    }).sort_values("period_days")