import polars as pl
from features.feature import FeatureProcol


class AmountAggregationFeature:
    def __init__(self, df: pl.DataFrame):
        self.df = df.sort("Time").with_columns(
            pl.from_epoch("Time", time_unit="s").alias("Time_dt")
        )

    def apply(self) -> pl.DataFrame:

        aggregations = []
        for window in ["1h", "6h", "24h"]:
            agg_df = self.df.group_by_dynamic(
                index_column="Time_dt", every="1s", period=window, closed="left"
            ).agg(
                pl.col("Amount").mean().alias(f"amount_mean_{window}"),
                pl.col("Amount").sum().alias(f"amount_sum_{window}"),
                pl.col("Amount").std().alias(f"amount_std_{window}"),
            )
            aggregations.append(agg_df)

        df_merged = self.df
        for agg_df in aggregations:
            df_merged = df_merged.join(agg_df, on="Time_dt", how="left")

        df_merged = df_merged.fill_null(0)

        epsilon = 1e-6
        df_with_ratios = df_merged.with_columns(
            [
                (pl.col("Amount") / (pl.col("amount_mean_1h") + epsilon)).alias(
                    "amount_to_mean_ratio_1h"
                ),
                (pl.col("Amount") / (pl.col("amount_mean_6h") + epsilon)).alias(
                    "amount_to_mean_ratio_6h"
                ),
                (pl.col("Amount") / (pl.col("amount_mean_24h") + epsilon)).alias(
                    "amount_to_mean_ratio_24h"
                ),
            ]
        )

        return df_with_ratios.drop("Time_dt")


def register() -> list[type[FeatureProcol]]:
    return [AmountAggregationFeature]
