import polars as pl


class IsNightFeature:
    def __init__(
        self, df: pl.DataFrame, night_start: int = 23, night_end: int = 8
    ) -> None:
        self.df = df
        self.night_start = night_start
        self.night_end = night_end

    def apply(self) -> pl.DataFrame:
        hour = (pl.col("Time") / 3600).floor().cast(pl.Int32).mod(24)
        is_night = ((hour.ge(self.night_start)) | (hour.lt(self.night_end))).cast(
            pl.UInt8
        )

        return self.df.with_columns([is_night.alias("isNight")])
