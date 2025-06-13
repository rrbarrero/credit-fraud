from typing import Protocol
import polars as pl


class DatasetCleanerProtocol(Protocol):
    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame: ...


class DatasetCleaner:
    @staticmethod
    def clean(df: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise ValueError("Input must be a Polars DataFrame")

        df_cleaned = df.drop_nulls()
        df_cleaned = df_cleaned.unique()

        return df_cleaned
