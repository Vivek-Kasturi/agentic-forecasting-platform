import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

from utils.logger import logger


class TimeSeriesIngestion:
    """
    Reusable ingestion pipeline for time-series datasets
    with schema validation + data contract enforcement.
    """

    def __init__(self):
        # -----------------------------
        # DATA CONTRACT (SCHEMA RULES)
        # -----------------------------
        self.required_columns = [
            "Store",
            "Date",
            "Weekly_Sales"
        ]

        self.expected_types = {
            "Store": "int",
            "Date": "datetime",
            "Weekly_Sales": "float"
        }

    def load_data(self):
        """
        Load Walmart dataset from KaggleHub.
        """

        file_path = "Walmart.csv"

        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "yasserh/walmart-dataset",
            file_path,
        )

        return df

    def validate_data(self, df):
        """
        Validate dataset against data contract.
        """

        logger.info("Validating dataset schema...")

        # 1. Check required columns
        missing_columns = [
            col for col in self.required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # 2. Check data types (light validation)
        for col, expected_type in self.expected_types.items():

            if col not in df.columns:
                continue

            if expected_type == "datetime":
                continue  # handled in cleaning step

            if expected_type == "int":
                if not pd.api.types.is_integer_dtype(df[col]):
                    logger.warning(f"Column {col} is not integer type")

            if expected_type == "float":
                if not pd.api.types.is_float_dtype(df[col]):
                    logger.warning(f"Column {col} is not float type")

        logger.info("Schema validation completed successfully.")

        return True

    def clean_data(self, df):
        """
        Basic cleaning and preprocessing.
        """

        logger.info("Cleaning dataset...")

        # Parse datetime
        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%d-%m-%Y"
        )

        # Remove duplicates
        df = df.drop_duplicates()

        # Drop missing values
        df = df.dropna()

        # Sort chronologically
        df = df.sort_values("Date")

        # Reset index
        df = df.reset_index(drop=True)

        logger.info("Data cleaning completed.")

        return df

    def run(self):
        """
        Execute full ingestion pipeline.
        """

        logger.info("Loading dataset...")

        df = self.load_data()

        self.validate_data(df)

        df = self.clean_data(df)

        logger.info("Ingestion pipeline completed successfully.")

        return df


if __name__ == "__main__":

    pipeline = TimeSeriesIngestion()

    df = pipeline.run()

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nDataset Shape:")
    print(df.shape)