from __future__ import annotations
from pathlib import Path
import requests
import numpy as np
import pandas as pd

OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"

# Official NASA OMNI2 column indices (0-based) and fill values
# Ref: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
OMNI_COLS = {
    0: "year",
    1: "doy",
    2: "hour",
    8: "b_scalar_nt",       # Scalar IMF B (nT), fill = 999.9
    12: "bz_gsm_nt",        # Bz GSM (nT), fill = 999.9
    21: "sigma_b_nt",       # RMS SD of field vector (nT), fill = 999.9
    23: "proton_density",   # Proton density (n/cc), fill = 999.9
    24: "sw_speed_kms",     # Bulk flow speed V_sw (km/s), fill = 9999.0
    28: "flow_pressure_npa",# Flow pressure (nPa), fill = 99.99
    39: "sunspot_r",        # Sunspot number R, fill = 999
    40: "dst_index_nt",     # Dst index (nT), fill = 99999
    50: "f10_7_index",      # F10.7 solar flux, fill = 999.9
}

FILL_THRESHOLDS = {
    "b_scalar_nt": 990.0,
    "bz_gsm_nt": 990.0,
    "sigma_b_nt": 990.0,
    "proton_density": 990.0,
    "sw_speed_kms": 9990.0,
    "flow_pressure_npa": 99.0,
    "sunspot_r": 990.0,
    "dst_index_nt": 99990.0,
    "f10_7_index": 990.0,
}


def fetch_omni_daily(
    start_date: str = "1996-01-01",
    end_date: str = "2024-12-31",
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Download NASA OMNI2 ASCII archive (if not cached) and aggregate to UTC daily means."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_file = raw_dir / "omni2_all_years.dat"

    if not local_file.exists():
        print(f"[OMNI] Downloading NASA GSFC OMNI2 archive from {OMNI2_URL} ...")
        resp = requests.get(OMNI2_URL, timeout=180)
        resp.raise_for_status()
        local_file.write_bytes(resp.content)
        print(f"[OMNI] Saved {local_file.stat().st_size / 1e6:.1f} MB to {local_file}")

    usecols = list(OMNI_COLS.keys())
    col_names = list(OMNI_COLS.values())

    df = pd.read_csv(
        local_file,
        sep=r"\s+",
        header=None,
        usecols=usecols,
        names=col_names,
        dtype=float,
    )

    start_yr = pd.Timestamp(start_date).year
    end_yr = pd.Timestamp(end_date).year
    df = df[(df["year"] >= start_yr) & (df["year"] <= end_yr)].copy()

    # Apply official NASA fill-value masks
    for col, thresh in FILL_THRESHOLDS.items():
        df.loc[df[col].abs() >= thresh, col] = np.nan

    # Construct UTC timestamp from year + day-of-year + hour
    timestamps = (
        pd.to_datetime(df["year"].astype(int).astype(str), format="%Y")
        + pd.to_timedelta(df["doy"].astype(int) - 1, unit="D")
        + pd.to_timedelta(df["hour"].astype(int), unit="h")
    )
    df.index = timestamps
    df = df.drop(columns=["year", "doy", "hour"])

    # Compute physical coupling covariates: convective electric field E_y = -V_sw * B_z * 1e-3 [mV/m]
    df["ey_convective_mvm"] = -df["sw_speed_kms"] * df["bz_gsm_nt"] * 1e-3
    # Heliospheric modulation driver parameter V_sw * B_t
    df["vb_product"] = df["sw_speed_kms"] * df["b_scalar_nt"]

    daily = df.resample("1D").mean()
    daily["missing_frac_omni"] = df["sw_speed_kms"].isna().resample("1D").mean()
    daily = daily.loc[start_date:end_date]
    return daily