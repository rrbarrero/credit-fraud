from typing import Protocol
import polars as pl


class FeatureProcol(Protocol):
    def __init__(self, df: pl.DataFrame) -> None: ...
    def apply(self) -> pl.DataFrame: ...
