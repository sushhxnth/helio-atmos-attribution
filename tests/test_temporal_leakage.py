import pandas as pd
from src.models.blocked_cv import SolarCycleBlockedSplitter


def test_zero_overlap_and_purge_buffer():
    idx = pd.date_range("1996-01-01", "2024-12-31", freq="1D")
    df = pd.DataFrame({"val": range(len(idx))}, index=idx)
    splitter = SolarCycleBlockedSplitter(purge_days=54)

    for train_idx, test_idx, _ in splitter.split(df):
        train_dates = df.index[train_idx]
        test_dates = df.index[test_idx]
        assert len(train_dates.intersection(test_dates)) == 0
        min_dist = abs((train_dates.values[:, None] - test_dates.values[None, :])).min()
        min_days = pd.Timedelta(min_dist).days
        assert min_days >= 54