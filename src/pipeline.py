from typing import Tuple, Type
import polars as pl
from dataclasses import dataclass
from balancers.balancer import BalancerProcol
from balancers.oversampling_balancer import OversamplingBalancer
from features.feature import FeatureProcol
from features.hour_of_day_feature import HourOfDayFeature
from features.time_hours_feature import TimeHoursFeature
from features.time_since_previous_feature import TimeSincePreviousFeature
from utils.data_utils import (
    DatasetCleaner,
    DatasetCleanerProtocol,
)
from utils.filesystem_utils import DatasetLoader, DatasetLoaderProtocol
from config import settings
from sklearn.model_selection import train_test_split


@dataclass
class DataPipeline:
    df: pl.DataFrame

    @property
    def X(self) -> pl.DataFrame:
        return self.df.drop(["Class"])

    @property
    def y(self) -> pl.Series:
        return self.df["Class"]

    def split(
        self, test_size: float = 0.3
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        return train_test_split(
            self.X.to_pandas(),
            self.y.to_pandas(),
            stratify=self.y.to_pandas(),
            test_size=test_size,
            random_state=42,
        )  # type: ignore


class PipelineBuilder:
    def __init__(
        self,
        dataset_loader: DatasetLoaderProtocol,
        dataframe_utils: DatasetCleanerProtocol,
    ):
        self.dataset_loader = dataset_loader
        self.dataframe_utils = dataframe_utils
        self.features: list[Type[FeatureProcol]] = []
        self.balancer: Type[BalancerProcol] | None = None
        self.dataset_path: str | None = None

    def _load(self) -> pl.DataFrame:
        if self.dataset_path is None:
            raise ValueError("Dataset path is not set. Use with_path() method first.")
        return self.dataset_loader.handle(self.dataset_path)

    def _clean(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.dataframe_utils.clean(df)

    def _apply_balancer(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.balancer:
            return self.balancer(df).apply()  # type: ignore
        return df

    def with_path(self, dataset_path: str) -> "PipelineBuilder":
        self.dataset_path = dataset_path
        return self

    def with_balancer(self, balancer: Type[BalancerProcol]) -> "PipelineBuilder":
        self.balancer = balancer
        return self

    def with_features(self, features: list[Type[FeatureProcol]]) -> "PipelineBuilder":
        self.features = features  # type: ignore
        return self

    def build(self) -> DataPipeline:
        df = self._load()
        df = self._clean(df)
        df = self._apply_balancer(df)
        for feature_class in self.features:
            df = feature_class(df).apply()

        return DataPipeline(df)

    @classmethod
    def default(cls):
        dataset_path = str(settings.data_path / "creditcard.csv.zip")
        features = [TimeHoursFeature, HourOfDayFeature, TimeSincePreviousFeature]
        return (
            PipelineBuilder(DatasetLoader, DatasetCleaner)
            .with_path(dataset_path)
            .with_features(features)
            .with_balancer(OversamplingBalancer)
            .build()
        )
