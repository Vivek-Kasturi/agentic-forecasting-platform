import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


def load_walmart_dataset():
    file_path = "Walmart.csv"

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "yasserh/walmart-dataset",
        file_path,
    )

    return df


if __name__ == "__main__":
    print("Loading dataset...")

    df = load_walmart_dataset()

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns)

    print("\nDataset Shape:")
    print(df.shape)