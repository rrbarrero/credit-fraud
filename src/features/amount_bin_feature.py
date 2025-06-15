import polars as pl
from typing import Optional, List


class AmountBinFeature:

    def __init__(
        self,
        df: pl.DataFrame,
        bins: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        self.df = df
        self.bins = bins if bins is not None else [10, 50, 100, 500, 1000]
        self.labels = (
            labels
            if labels is not None
            else ["<10", "10-50", "50-100", "100-500", "500-1000", ">=1000"]
        )
        if len(self.labels) != len(self.bins) + 1:
            raise ValueError("Wrong number of labels, they have to be len(bins) + 1")

    def apply(self) -> pl.DataFrame:
        amt = pl.col("Amount")

        expr = pl.when(amt < self.bins[0]).then(pl.lit(self.labels[0]))

        for lower, upper, label in zip(self.bins, self.bins[1:], self.labels[1:-1]):
            expr = expr.when((amt >= lower) & (amt < upper)).then(pl.lit(label))

        expr = expr.when(amt >= self.bins[-1]).then(pl.lit(self.labels[-1]))

        expr = expr.otherwise(pl.lit("unknown"))

        return self.df.with_columns([expr.alias("amountBin")])
