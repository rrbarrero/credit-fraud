import polars as pl
from features.feature import FeatureProcol


class TimeHoursFeature:
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def apply(self) -> pl.DataFrame:
        df = self.df.with_columns(((pl.col("Time") / 3600) % 24).alias("TimeHours"))
        return df


def register() -> list[type[FeatureProcol]]:
    return [TimeHoursFeature]
