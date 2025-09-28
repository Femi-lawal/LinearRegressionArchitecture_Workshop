import numpy as np
import pandas as pd
from typing import List, Tuple

# 🟦 Identify contiguous runs in a boolean mask
def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Given a boolean mask (array of True/False), return start and end indices of contiguous True runs.
    Example: [False, True, True, False, True] -> [(1, 2), (4, 4)]
    """
    runs, in_run, start = [], False, 0
    for i, val in enumerate(mask):
        if val and not in_run:
            in_run, start = True, i
        elif not val and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


# 🟦 Convert time column to seconds relative to first timestamp
def to_seconds(tseries: pd.Series) -> pd.Series:
    """
    Convert a datetime column/series to seconds since the first entry.
    """
    t = pd.to_datetime(tseries)
    return (t - t.iloc[0]).dt.total_seconds()


# 🟦 Compute median delta (step size) in seconds
def median_dt_seconds(tseries: pd.Series) -> float:
    """
    Compute median time step (in seconds) for a datetime column/series.
    """
    t = pd.to_datetime(tseries)
    dt = t.diff().dt.total_seconds().dropna()
    return float(dt.median() if len(dt) else 1.0)
