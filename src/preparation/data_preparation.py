import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreparation:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataPreparation with raw DataFrame.
        Args:
            data (pd.DataFrame): Raw input data (DB snapshot or CSV).
        """
        self.data = data.copy()
        self.scaler_std = None
        self.scaler_mm = None

    def clean_data(self):
        """
        Cleans the dataset:
        - Converts timestamps to datetime
        - Drops duplicate rows
        - Handles missing values (forward-fill, then drop remaining)
        - Removes impossible negative values (if currents cannot be negative)
        """
        # Standardize timestamp column
        if "Time" in self.data.columns:
            self.data["Time"] = pd.to_datetime(self.data["Time"], errors="coerce")

        # Drop duplicates
        self.data.drop_duplicates(inplace=True)

        # Forward-fill missing values, then drop if still missing
        self.data.fillna(method="ffill", inplace=True)
        self.data.dropna(inplace=True)

        # Remove negative currents (if domain knowledge says they’re invalid)
        axis_cols = [c for c in self.data.columns if c.startswith("axis_")]
        for col in axis_cols:
            self.data.loc[self.data[col] < 0, col] = 0.0

    def transform_data(self):
        """
        Transforms the dataset:
        - Adds elapsed time column in seconds
        - Normalizes axis columns with StandardScaler and MinMaxScaler
        """
        # Add elapsed time in seconds if Time column exists
        if "Time" in self.data.columns:
            t0 = self.data["Time"].iloc[0]
            self.data["t_sec"] = (self.data["Time"] - t0).dt.total_seconds()

        # Identify axis columns
        axis_cols = [c for c in self.data.columns if c.startswith("axis_")]

        # Fit scalers on axis data
        self.scaler_std = StandardScaler()
        self.scaler_mm = MinMaxScaler()

        self.data[[f"{c}_std" for c in axis_cols]] = self.scaler_std.fit_transform(self.data[axis_cols])
        self.data[[f"{c}_mm" for c in axis_cols]] = self.scaler_mm.fit_transform(self.data[axis_cols])

    def get_prepared_data(self) -> pd.DataFrame:
        """
        Executes cleaning + transformation pipeline and returns prepared DataFrame.
        """
        self.clean_data()
        self.transform_data()
        return self.data
