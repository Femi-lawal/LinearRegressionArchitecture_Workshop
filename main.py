import os
import yaml
import pandas as pd

from extraction import PostgresExtractor, StreamSimulator
from preparation import DataPreparation
from models import Trainer, SyntheticDataGenerator, AnomalyDetector
from utils import to_seconds

CONFIG_PATH = "configs/experiment_config.yaml"


def load_config(path: str):
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1️⃣ Load experiment configuration
    cfg = load_config(CONFIG_PATH)
    outdir = cfg.get("outdir", "experiments")
    os.makedirs(outdir, exist_ok=True)

    # 2️⃣ Load training data from DB
    db = PostgresExtractor(cfg["db"])
    df_train = db.fetch_training_snapshot(
        table=cfg["training"]["table"],
        limit=cfg["training"]["limit"]
    )

    # 3️⃣ Data preparation
    prep = DataPreparation(df_train)
    df_train_clean = prep.get_prepared_data()

    # 4️⃣ Train regression models + thresholds
    trainer = Trainer(outdir=outdir)
    trainer.fit(df_train_clean)
    trainer.fit_scalers(df_train_clean)

    # 5️⃣ Generate synthetic test data
    gen = SyntheticDataGenerator(trainer.models, trainer.thresholds, outdir=outdir)
    df_test = gen.generate(
        t_start=df_train_clean["Time"].iloc[-1],
        n_rows=cfg["testing"]["synthetic_rows"],
        dt_s=cfg["testing"]["dt_seconds"]
    )

    # 6️⃣ Stream data locally
    sim = StreamSimulator(os.path.join(outdir, "synthetic_test.csv"))
    df_streamed = sim.run(max_steps=cfg["testing"]["stream_steps"])

    # 7️⃣ Detect anomalies
    detector = AnomalyDetector(trainer.models, trainer.thresholds, outdir=outdir)
    detector.detect(df_streamed)
    ev_path = detector.save()
    print(f"✅ Events saved to {ev_path}")

    print("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
