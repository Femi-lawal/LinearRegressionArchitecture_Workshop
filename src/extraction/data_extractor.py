class DataExtractor:
    def __init__(self, source_type, config):
        self.source_type = source_type
        self.config = config

    def load(self):
        if self.source_type == "csv":
            from .CSVExtractor import CSVExtractor
            return CSVExtractor(self.config["path"]).load()
        elif self.source_type == "db":
            from .PostgresExtractor import PostgresExtractor
            return PostgresExtractor(self.config).fetch_table(self.config["table"])
        else:
            raise ValueError(f"Unknown source_type {self.source_type}")
