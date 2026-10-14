from __future__ import annotations
import io
import re
from pathlib import Path
import requests
import numpy as np
import pandas as pd


OULU_PUBLIC_URL = "https://cosmicrays.oulu.fi/readme.html"
NMDB_NEST_URL = "https://www.nmdb.eu/nest/draw_graph.php"


def _fetch_nmdb_station_chunk(station: str, start_year: int, end_year: int) -> pd.Series:
    """Query NMDB NEST ASCII output for a single station and year range."""
    params = {
        "formchk": "1",
        "stations[]": station,
        "tabchoice": "revori",
        "dtype": "corr_for_efficiency",
        "tresolution": "1440",  # Daily (1440 min)
        "yunits": "0",
        "date_choice": "bydate",
        "start_year": str(start_year),
        "start_month": "01",
        "start_day": "01",
        "start_hour": "00",
        "start_min": "00",
        "end_year": str(end_year),
        "end_month": "12",
        "end_day": "31",
        "end_hour": "23",
        "end_min": "59",
        "output": "ascii",
    }
    headers = {"User-Agent": "helio-atmos-research-pipeline/1.0"}
    resp = requests.get(NMDB_NEST_URL, params=params, headers=headers, timeout=90)
    resp.raise_for_status()

    lines = []
    for line in resp.text.splitlines():
        line_str = line.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}", line_str):
            parts = [p.strip() for p in re.split(r"[;\s]+", line_str) if p.strip()]
            if len(parts) >= 3:
                lines.append((parts[0], parts[-1]))
            elif len(parts) == 2:
                lines.append((parts[0], parts[1]))

    if not lines:
        raise RuntimeError(f"No valid ASCII rows returned from NMDB for {station}")

    df = pd.DataFrame(lines, columns=["date", "count"])
    df["date"] = pd.to_datetime(df["date"])
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df.loc[df["count"] <= 0, "count"] = np.nan
    s = df.set_index("date")["count"].resample("1D").mean()
    return s


def fetch_neutron_monitors_daily(
    start_date: str = "1996-01-01",
    end_date: str = "2024-12-31",
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
  
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "nmdb_daily_counts.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df.loc[start_date:end_date]

    start_yr = pd.Timestamp(start_date).year
    end_yr = pd.Timestamp(end_date).year

    stations = ["OULU", "MOSC"]
    station_series = {}

    for st in stations:
        print(f"[NMDB] Fetching pressure-corrected daily counts for station {st} ({start_yr}-{end_yr})...")
        chunks = []
        # Query in 10-year intervals to respect NMDB server timeout limits
        for y0 in range(start_yr, end_yr + 1, 10):
            y1 = min(y0 + 9, end_yr)
            try:
                s_chunk = _fetch_nmdb_station_chunk(st, y0, y1)
                chunks.append(s_chunk)
            except Exception as exc:
                print(f"[NMDB] Warning on {st} ({y0}-{y1}): {exc}")
        if chunks:
            full_s = pd.concat(chunks).sort_index()
            full_s = full_s[~full_s.index.duplicated(keep="first")]
            station_series[st] = full_s

    idx = pd.date_range(start_date, end_date, freq="1D")
    out = pd.DataFrame(index=idx)

    for st, s in station_series.items():
        aligned = s.reindex(idx)
        out[f"nm_{st.lower()}_raw"] = aligned
        # Normalize to Solar Cycle 23/24 quiet minimum baseline (2008-2009)
        baseline = aligned.loc["2008-01-01":"2009-12-31"].median()
        if pd.isna(baseline) or baseline <= 0:
            baseline = aligned.median()
        out[f"nm_{st.lower()}_norm"] = aligned / baseline
        out[f"nm_{st.lower()}_pct_anom"] = (aligned / baseline - 1.0) * 100.0

    out.to_parquet(cache_path)
    return out.loc[start_date:end_date]