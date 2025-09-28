import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .axis_model import AxisModel
from .residual_analyzer import ResidualAnalyzer
from visualization.visualizer import Visualizer  # assuming visualization/visualizer.py

class Trainer:
    def __init__(self, db_handler, outdir="pm_outputs"):
        self.db = db_handler
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.models = {}
        self.thresholds = {}
        self.scaler_std = None
        self.scaler_mm = None

    def load_training(self, table="robot_readings", limit=50000) -> pd.DataFrame:
        df = self.db.fetch_training_snapshot(table=table, limit=limit)
        df["t_sec"] = pd.to_datetime(df["Time"])
        df["t_sec"] = (df["t_sec"] - df["t_sec"].iloc[0]).dt.total_seconds()
        return df

    def fit(self, df_train: pd.DataFrame):
        t = df_train["t_sec"].to_numpy()
        dt_s = float(df_train["t_sec"].diff().median()) or 1.0
        analyzer = ResidualAnalyzer(dt_s)
        viz = Visualizer(self.outdir)
        coeffs = []

        for axis in [c for c in df_train.columns if c.startswith("axis_") and int(c.split("_")[1]) <= 8]:
            y = df_train[axis].to_numpy()
            m = AxisModel(axis).fit(t, y)
            th = analyzer.discover(m, t, y)

            # plots
            yhat = m.predict(t)
            viz.plot_regression(axis, t, y, yhat)
            resid = m.residuals(t, y)
            viz.plot_residuals(axis, t, resid, th)

            self.models[axis] = m
            self.thresholds[axis] = th
            coeffs.append(m.fit_report(t, y).__dict__)

        pd.DataFrame(coeffs).to_csv(os.path.join(self.outdir, "model_coeffs.csv"), index=False)
        pd.DataFrame([vars(th) for th in self.thresholds.values()]).to_csv(
            os.path.join(self.outdir, "thresholds.csv"), index=False
        )

    def fit_scalers(self, df_train: pd.DataFrame):
        axes = [c for c in df_train.columns if c.startswith("axis_") and int(c.split("_")[1]) <= 8]
        self.scaler_std = StandardScaler().fit(df_train[axes].values)
        self.scaler_mm = MinMaxScaler().fit(df_train[axes].values)

    def transform(self, df: pd.DataFrame):
        axes = [c for c in df.columns if c.startswith("axis_") and int(c.split("_")[1]) <= 8]
        std = pd.DataFrame(self.scaler_std.transform(df[axes]), columns=axes)
        mm = pd.DataFrame(self.scaler_mm.transform(df[axes]), columns=axes)
        return std, mm
