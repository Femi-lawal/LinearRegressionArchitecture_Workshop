import pandas as pd


class FeatureEngineer:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific engineered features."""

        # Example: rolling mean per axis
        for col in df.select_dtypes(include="number").columns:
            df[f"{col}_rollmean"] = df[col].rolling(window=5, min_periods=1).mean()

        # Example: squared values (detect higher energy usage)
        for col in df.select_dtypes(include="number").columns:
            df[f"{col}_squared"] = df[col] ** 2

        return df
