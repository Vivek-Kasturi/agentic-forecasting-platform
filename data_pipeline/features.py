import numpy as np
import pandas as pd


# =========================
# DATETIME FEATURES
# =========================
def create_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    df["year"] = df[datetime_col].dt.year
    df["month"] = df[datetime_col].dt.month
    df["day"] = df[datetime_col].dt.day
    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["week_of_year"] = df[datetime_col].dt.isocalendar().week.astype(int)
    df["quarter"] = df[datetime_col].dt.quarter

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


# =========================
# PANEL-SAFE LAGS
# =========================
def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    lags: list[int]
) -> pd.DataFrame:

    df = df.copy()

    df = df.sort_values(group_cols + ["Date"])

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = (
            df.groupby(group_cols)[target_col].shift(lag)
        )

    return df


# =========================
# PANEL-SAFE ROLLING
# =========================
def create_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    windows: list[int]
) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values(group_cols + ["Date"])

    for window in windows:

        df[f"{target_col}_rolling_mean_{window}"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda x: x.rolling(window).mean())
        )

        df[f"{target_col}_rolling_std_{window}"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda x: x.rolling(window).std())
        )

    return df


# =========================
# SEASONALITY FEATURES
# =========================
def create_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# =========================
# FULL PIPELINE
# =========================
def build_feature_pipeline(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: str,
    group_cols: list[str]
) -> pd.DataFrame:

    df = df.copy()

    # Step 1: datetime
    df = create_datetime_features(df, datetime_col)

    # Step 2: lag features (panel-safe)
    df = create_lag_features(
        df,
        target_col,
        group_cols,
        lags=[1, 2, 7, 14]
    )

    # Step 3: rolling features (panel-safe)
    df = create_rolling_features(
        df,
        target_col,
        group_cols,
        windows=[3, 7, 14]
    )

    # Step 4: seasonality
    df = create_seasonality_features(df)

    # Step 5: cleanup
    df = df.dropna().reset_index(drop=True)

    return df