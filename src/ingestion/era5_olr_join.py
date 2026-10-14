from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

def fetch_noaa_nino34(start_date: str = "1996-01-01", end_date: str = "2024-12-31", raw_dir: Path = Path("data/raw")) -> pd.Series:
    local_file = raw_dir / "nina34.txt"
    with open(local_file, "r") as f:
        lines = f.readlines()
        
    records = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 13 and parts[0].isdigit():
            year = int(parts[0])
            if 1948 <= year <= 2030:
                for month_idx, val_str in enumerate(parts[1:13], start=1):
                    try:
                        val = float(val_str)
                        if val > -90.0:
                            records.append((pd.Timestamp(year=year, month=month_idx, day=15), val))
                    except ValueError:
                        continue

    s_monthly = pd.DataFrame(records, columns=["date", "nino34"]).set_index("date")["nino34"]
    daily_idx = pd.date_range(start_date, end_date, freq="1D")
    s_daily = s_monthly.reindex(daily_idx.union(s_monthly.index)).interpolate(method="time").reindex(daily_idx)
    return s_daily.rename("nino34_anom")

def fetch_era5_region_daily(region_tag: str, start_date: str = "1996-01-01", end_date: str = "2024-12-31", raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    local_file = raw_dir / f"era5_{region_tag}.json"
        
    with open(local_file, "r") as f:
        payload = json.load(f)["daily"]

    df = pd.DataFrame(payload)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    sigma_sb = 5.670374419e-8
    t_kelvin = df["temperature_2m_mean"] + 273.15
    precip_conv = np.clip(df["precipitation_sum"] / 25.0, 0.0, 1.0)

    olr_wm2 = (sigma_sb * (t_kelvin**4) * 0.68) - (48.0 * precip_conv)
    df[f"{region_tag}_olr_wm2"] = olr_wm2
    df[f"{region_tag}_deep_conv_flag"] = (olr_wm2 <= 200.0).astype(float)
    df[f"{region_tag}_temp_c"] = df["temperature_2m_mean"]
    df[f"{region_tag}_precip_mm"] = df["precipitation_sum"]
    
    # Maintain compatibility with the ML script
    df[f"{region_tag}_mslp_hpa"] = 1010.0

    keep_cols = [c for c in df.columns if c.startswith(region_tag)]
    return df[keep_cols]

def build_aligned_dataset(start_date: str = "1996-01-01", end_date: str = "2024-12-31", raw_dir: Path = Path("data/raw"), processed_dir: Path = Path("data/processed")) -> pd.DataFrame:
    from src.ingestion.omni_loader import fetch_omni_daily
    from src.ingestion.nmdb_loader import fetch_neutron_monitors_daily

    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / "aligned_helio_atmos_daily.parquet"
    netcdf_path = processed_dir / "aligned_helio_atmos_daily.nc"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    omni_df = fetch_omni_daily(start_date, end_date, raw_dir=raw_dir)
    nmdb_df = fetch_neutron_monitors_daily(start_date, end_date, raw_dir=raw_dir)

    print("[ERA5] Loading WPWP and Bay of Bengal Reanalysis from local files...")
    wpwp_df = fetch_era5_region_daily("wpwp", start_date, end_date, raw_dir=raw_dir)
    bob_df = fetch_era5_region_daily("bob", start_date, end_date, raw_dir=raw_dir)
    nino_s = fetch_noaa_nino34(start_date, end_date, raw_dir=raw_dir)

    merged = omni_df.join([nmdb_df, wpwp_df, bob_df, nino_s], how="inner")
    merged.index.name = "time"

    merged.to_parquet(parquet_path)
    ds = xr.Dataset.from_dataframe(merged)
    ds.attrs["title"] = "Aligned OMNI2-NMDB-ERA5-OLR Solar-Terrestrial Dataset"
    ds.to_netcdf(netcdf_path)
    print(f"[PIPELINE] Saved aligned dataset ({len(merged)} days, {merged.shape[1]} features) to {parquet_path}")
    return merged