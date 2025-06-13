from typing import Protocol
import polars as pl
from imblearn.over_sampling import SMOTE
import pandas as pd


class DatasetCleanerProtocol(Protocol):
    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame: ...

    @staticmethod
    def balance(df: pl.DataFrame) -> pl.DataFrame: ...


class DatasetCleaner:

    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise ValueError("Input must be a Polars DataFrame")

        df_cleaned = df.drop_nulls()
        df_cleaned = df_cleaned.unique()

        return df_cleaned

    @staticmethod
    def balance(df: pl.DataFrame) -> pl.DataFrame:
        df = df.unique()

        df_pd = df.to_pandas()
        X = df_pd.drop(columns=["Class"])
        y = df_pd["Class"]

        smote = SMOTE(k_neighbors=1, random_state=42)
        result = smote.fit_resample(X, y)
        if len(result) == 2:
            X_resampled, y_resampled = result
        else:
            X_resampled, y_resampled, _ = result

        df_resampled = pl.DataFrame(pd.DataFrame(X_resampled, columns=X.columns))
        df_resampled = df_resampled.with_columns(pl.Series("Class", y_resampled))

        return df_resampled
