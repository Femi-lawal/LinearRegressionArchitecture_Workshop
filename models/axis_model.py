import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

@dataclass
class AxisCoeffs:
    axis: str
    slope: float
    intercept: float
    r2: float

class AxisModel:
    def __init__(self, axis: str):
        self.axis = axis
        self.model = LinearRegression()
        self.slope = None
        self.intercept = None
        self.resid_mu = None
        self.resid_sigma = None

    def fit(self, t_sec: np.ndarray, y: np.ndarray):
        X = t_sec.reshape(-1, 1)
        self.model.fit(X, y)
        self.slope = float(self.model.coef_[0])
        self.intercept = float(self.model.intercept_)
        return self

    def predict(self, t_sec: np.ndarray) -> np.ndarray:
        return self.model.predict(t_sec.reshape(-1, 1))

    def residuals(self, t_sec: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y - self.predict(t_sec)

    def fit_report(self, t_sec: np.ndarray, y: np.ndarray) -> AxisCoeffs:
        yhat = self.predict(t_sec)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) if len(y) else 0.0
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        return AxisCoeffs(self.axis, self.slope, self.intercept, r2)
