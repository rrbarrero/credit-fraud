import polars as pl


class TimeSincePreviousFeature:

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def apply(self) -> pl.DataFrame:

        df = self.df.sort("Time")

        df = df.with_columns(
            [
                pl.col("Time")
                .diff(1)
                .fill_null(0)
                .cast(pl.Float64)
                .alias("TimeSincePrevSec")
            ]
        )

        df = df.with_columns(
            [(pl.col("Time") % 86400).cast(pl.Float64).alias("TxnTimeSec")]
        )

        return df
