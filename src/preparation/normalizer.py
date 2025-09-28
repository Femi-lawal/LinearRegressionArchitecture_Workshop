import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Normalizer:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
