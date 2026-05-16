import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from config.settings import TIMESTAMP_COLUMN
from utils.logger import logger


def load_walmart_dataset():
    file_path = "Walmart.csv"

    logger.info("Loading Walmart dataset from Kaggle")

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "yasserh/walmart-dataset",
        file_path,
    )

    # ---- Data preprocessing ----
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COLUMN])

    # Sort by full panel structure (IMPORTANT FIX)
    sort_cols = [col for col in ["Store", "Dept", TIMESTAMP_COLUMN] if col in df.columns]
    df = df.sort_values(sort_cols)

    # Keep timestamp column (DO NOT set as index in multi-series dataset)
    logger.info(f"Dataset loaded successfully. Shape={df.shape}")

    return df


if __name__ == "__main__":
    logger.info("Running dataset loader standalone")

    df = load_walmart_dataset()

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns)

    print("\nDataset Shape:")
    print(df.shape)