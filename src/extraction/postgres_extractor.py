import psycopg2
import pandas as pd

class PostgresExtractor:
    def __init__(self, db_config):
        self.db_config = db_config

    def fetch_table(self, table: str, limit: int = 10000) -> pd.DataFrame:
        with psycopg2.connect(**self.db_config) as conn:
            query = f"SELECT * FROM {table} LIMIT {limit};"
            return pd.read_sql(query, conn)
