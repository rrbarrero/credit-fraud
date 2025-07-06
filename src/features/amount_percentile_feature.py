import polars as pl
from features.feature import FeatureProcol


class AmountPercentileFeature:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def apply(self) -> pl.DataFrame:
        original_index_col = "__original_index__"
        df_with_index = self.df.with_row_index(name=original_index_col)
        df_sorted_with_index = df_with_index.sort("Amount")

        n_rows = df_sorted_with_index.height

        if n_rows <= 1:
            return self.df.with_columns(pl.lit(0.0).alias("amount_percentile"))

        df_with_percentile_and_index = df_sorted_with_index.with_columns(
            ((pl.col("Amount").rank(method="average") - 1) / (n_rows - 1)).alias(
                "amount_percentile"
            )
        )

        final_df = df_with_index.join(
            df_with_percentile_and_index.select(
                [original_index_col, "amount_percentile"]
            ),
            on=original_index_col,
            how="left",
        ).drop(original_index_col)

        return final_df


def register() -> list[type[FeatureProcol]]:
    return [AmountPercentileFeature]
