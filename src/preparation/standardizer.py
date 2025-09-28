import pandas as pd
from sklearn.preprocessing import StandardScaler


class Standardizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
