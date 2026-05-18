import pandas as pd

from data_pipeline.load_dataset import load_walmart_dataset
from data_pipeline.validation import run_validation
from data_pipeline.features import build_feature_pipeline
from utils.time_split import panel_time_split, validate_temporal_order
from utils.logger import logger


def main():

    logger.info("🚀 Agentic Forecasting Platform initialized")

    # Step 1: Load data
    df = load_walmart_dataset()

    logger.info(f"Dataset loaded with shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Step 2: Validate
    run_validation(df)

    # Step 3: Safe datetime conversion
    df["Date"] = pd.to_datetime(df["Date"])

    # Step 4: Dynamic grouping (FIXED)
    group_cols = []

    if "Store" in df.columns:
        group_cols.append("Store")

    if "Dept" in df.columns:
        group_cols.append("Dept")

    group_cols.append("Date")

    df = df.sort_values(group_cols).reset_index(drop=True)

    # Step 5: Feature engineering (SAFE)
    feature_group_cols = []

    if "Store" in df.columns:
        feature_group_cols.append("Store")

    if "Dept" in df.columns:
        feature_group_cols.append("Dept")

    df = build_feature_pipeline(
        df=df,
        target_col="Weekly_Sales",
        datetime_col="Date",
        group_cols=feature_group_cols
    )

    logger.info(f"Feature engineering complete: {df.shape}")

    # Step 6: Split (SAFE)
    train_df, test_df = panel_time_split(
        df,
        group_cols=feature_group_cols,
        time_col="Date",
        train_ratio=0.8
    )

    # Step 7: Validate
    validate_temporal_order(train_df, test_df, "Date")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()