import pandas as pd
from utils.logger import logger


def panel_time_split(
    df: pd.DataFrame,
    group_cols: list[str],
    time_col: str,
    train_ratio: float = 0.8
):

    logger.info("Performing panel-aware chronological split")

    train_parts = []
    test_parts = []

    for _, group in df.groupby(group_cols):

        group = group.sort_values(time_col)

        split_idx = int(len(group) * train_ratio)

        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    return train_df, test_df


def validate_temporal_order(train_df, test_df, time_col):

    logger.info("Validating temporal integrity")

    train_max = train_df[time_col].max()
    test_min = test_df[time_col].min()

    if train_max >= test_min:
        logger.warning(
            "Temporal overlap detected (expected in panel datasets)"
        )
    else:
        logger.info("Temporal validation passed")