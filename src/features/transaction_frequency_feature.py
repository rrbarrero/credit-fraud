import polars as pl
from features.feature import FeatureProcol


class TransactionFrequencyFeature:
    def __init__(self, df: pl.DataFrame):
        self.df = df.sort("Time").with_columns(
            pl.from_epoch("Time", time_unit="s").alias("Time_dt")
        )

    def apply(self) -> pl.DataFrame:

        df_1h = self.df.group_by_dynamic(
            index_column="Time_dt", every="1s", period="1h", closed="left"
        ).agg(pl.count().alias("transactions_last_1h"))

        df_6h = self.df.group_by_dynamic(
            index_column="Time_dt", every="1s", period="6h", closed="left"
        ).agg(pl.count().alias("transactions_last_6h"))

        df_24h = self.df.group_by_dynamic(
            index_column="Time_dt", every="1s", period="24h", closed="left"
        ).agg(pl.count().alias("transactions_last_24h"))

        df_merged = self.df.join(df_1h, on="Time_dt", how="left")
        df_merged = df_merged.join(df_6h, on="Time_dt", how="left")
        df_merged = df_merged.join(df_24h, on="Time_dt", how="left")

        return df_merged.drop("Time_dt").fill_null(0)


def register() -> list[type[FeatureProcol]]:
    return [TransactionFrequencyFeature]
