from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SolarCycleBlock:
    name: str
    start_date: str
    end_date: str


SOLAR_CYCLE_BLOCKS: List[SolarCycleBlock] = [
    SolarCycleBlock("SC23_Rise_Max", "1996-01-01", "2002-12-31"),
    SolarCycleBlock("SC23_Decline", "2003-01-01", "2008-11-30"),
    SolarCycleBlock("SC24_Full", "2008-12-01", "2019-11-30"),
    SolarCycleBlock("SC25_Rise_Max", "2019-12-01", "2024-12-31"),
]


class SolarCycleBlockedSplitter:
    """
    Hold out contiguous Solar Cycle blocks with a `purge_days` buffer window
    on both boundaries to eliminate lag-feature and AR(1) autocorrelation leakage.
    """

    def __init__(self, blocks: List[SolarCycleBlock] = SOLAR_CYCLE_BLOCKS, purge_days: int = 54):
        self.blocks = blocks
        self.purge_days = pd.Timedelta(days=purge_days)

    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a pd.DatetimeIndex.")

        indices = np.arange(len(df))
        for block in self.blocks:
            t_start = pd.Timestamp(block.start_date)
            t_end = pd.Timestamp(block.end_date)

            test_mask = (df.index >= t_start) & (df.index <= t_end)
            purge_mask = (df.index >= (t_start - self.purge_days)) & (
                df.index <= (t_end + self.purge_days)
            )
            train_mask = ~purge_mask

            if test_mask.sum() > 0 and train_mask.sum() > 0:
                yield indices[train_mask], indices[test_mask], block.name


def detect_forbush_decrease_events(
    nm_pct_anom: pd.Series,
    drop_threshold_pct: float = -3.5,
    min_separation_days: int = 20,
) -> pd.DataFrame:
    """
    Identify rapid transient Forbush Decrease (FD) events where cosmic-ray count rate
    drops by more than `drop_threshold_pct` over a 5-day rolling onset window.
    """
    short_diff = nm_pct_anom - nm_pct_anom.rolling(7, min_periods=3).max()
    candidates = short_diff[short_diff <= drop_threshold_pct].dropna()

    events = []
    last_date = pd.Timestamp("1900-01-01")
    for dt, val in candidates.items():
        if (dt - last_date).days >= min_separation_days:
            window = short_diff.loc[dt : dt + pd.Timedelta(days=7)]
            trough_dt = window.idxmin()
            trough_val = float(window.min())
            events.append({
                "onset_date": dt,
                "trough_date": trough_dt,
                "drop_magnitude_pct": trough_val,
            })
            last_date = trough_dt

    return pd.DataFrame(events)