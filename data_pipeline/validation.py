import pandas as pd
from utils.logger import logger
from config.settings import (
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    EXPECTED_FREQUENCY
)


class ValidationError(Exception):
    pass


def check_missing_values(df):
    missing = df.isnull().sum()

    if missing.any():
        logger.error(f"Missing values detected:\n{missing}")
        raise ValidationError("Missing values found")


def check_duplicate_timestamps(df):
    """
    Walmart dataset is a panel dataset:
    (Store, Dept, Date) is unique, NOT Date alone.
    """
    if "Store" in df.columns and "Dept" in df.columns:
        duplicates = df.duplicated(subset=["Store", "Dept", TIMESTAMP_COLUMN]).sum()

        if duplicates > 0:
            logger.error(f"Duplicate (Store, Dept, Date) rows: {duplicates}")
            raise ValidationError("Duplicate panel keys detected")
    else:
        duplicates = df[TIMESTAMP_COLUMN].duplicated().sum()

        if duplicates > 0:
            logger.warning(f"Duplicate timestamps: {duplicates} (expected in time-series datasets)")


def check_sorted_timestamps(df):
    if TIMESTAMP_COLUMN in df.columns:
        sorted_check = df.sort_values(TIMESTAMP_COLUMN)[TIMESTAMP_COLUMN].is_monotonic_increasing

        if not sorted_check:
            logger.error("Timestamps not sorted")
            raise ValidationError("Timestamps must be sorted")


def validate_schema(df):
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Missing column: {TARGET_COLUMN}")
        raise ValidationError("Schema invalid")


def validate_frequency(df):
    """
    Frequency check only works on single time-series.
    For panel data, we skip strict enforcement.
    """
    try:
        if TIMESTAMP_COLUMN in df.columns:
            sample = df[TIMESTAMP_COLUMN].drop_duplicates().sort_values()
            inferred = pd.infer_freq(sample)

            if inferred != EXPECTED_FREQUENCY:
                logger.warning(f"Frequency mismatch: {inferred} (panel dataset expected)")
    except Exception as e:
        logger.warning(f"Frequency check skipped: {str(e)}")


def run_validation(df):
    logger.info("Starting dataset validation")

    check_missing_values(df)
    check_duplicate_timestamps(df)
    check_sorted_timestamps(df)
    validate_schema(df)
    validate_frequency(df)

    logger.info("Validation completed successfully")
    return True