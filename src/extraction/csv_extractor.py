import pandas as pd

class CSVExtractor:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath)
