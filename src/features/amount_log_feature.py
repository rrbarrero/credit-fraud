import polars as pl
from features.feature import FeatureProcol


class AmountLogFeature:
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def apply(self) -> pl.DataFrame:
        return self.df.with_columns([(pl.col("Amount") + 1).log().alias("amountLog")])


def register() -> list[type[FeatureProcol]]:
    return [AmountLogFeature]
