import numpy as np
from dataclasses import dataclass
from utils.helpers import contiguous_runs  # assuming helpers.py is in a utils folder

@dataclass
class AxisThresholds:
    axis: str
    z_alert: float
    z_error: float
    MinC: float
    MaxC: float
    T_seconds: float
    dt_s: float
    resid_mu: float
    resid_sigma: float

class ResidualAnalyzer:
    def __init__(self, dt_s: float, z_alert=1.65, z_error=2.57):
        self.dt_s = dt_s
        self.z_alert = z_alert
        self.z_error = z_error

    def discover(self, axis_model, t_sec: np.ndarray, y: np.ndarray) -> AxisThresholds:
        resid = axis_model.residuals(t_sec, y)
        mu = float(np.mean(resid))
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0

        # store back into the model
        axis_model.resid_mu = mu
        axis_model.resid_sigma = sigma

        minC = self.z_alert * sigma
        maxC = self.z_error * sigma

        # estimate sustained anomaly duration
        mask = resid - mu > minC
        runs = contiguous_runs(mask)
        lengths = [j - i + 1 for i, j in runs] or [3]
        pts_thresh = int(np.quantile(lengths, 0.95))
        T_seconds = max(3, pts_thresh) * self.dt_s

        return AxisThresholds(axis_model.axis,
                              self.z_alert, self.z_error,
                              float(minC), float(maxC),
                              float(T_seconds), self.dt_s,
                              mu, sigma)
