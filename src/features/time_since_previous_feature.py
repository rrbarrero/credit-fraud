import polars as pl


class TimeSincePreviousFeature:
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def apply(self) -> pl.DataFrame:
        df = (
            self.df.lazy()
            .with_columns(pl.duration(seconds=pl.col("Time")).alias("TxnTime"))
            .sort("TxnTime")
            .with_columns(
                pl.col("TxnTime")
                .diff()
                .dt.total_seconds()
                .fill_null(0)
                .alias("TimeSincePrevSec")
            )
            .collect()
        )
        return df
