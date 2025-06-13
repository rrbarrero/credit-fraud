from typing import Type
import polars as pl
from dataclasses import dataclass
from balancers.balancer import BalancerProcol
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
from balancers.oversampling_balancer import OversamplingBalancer


@dataclass
class Pipeline:
    df: pl.DataFrame


class PipelineBuilder:
    def __init__(
        self,
        dataset_loader: DatasetLoaderProtocol,
        dataframe_utils: DatasetCleanerProtocol,
    ):
        self.dataset_loader = dataset_loader
        self.dataframe_utils = dataframe_utils
        self.df = None
        self.features: list[FeatureProcol] = []

    def load(self, file_path: str) -> "PipelineBuilder":
        self.df = self.dataset_loader.handle(file_path)
        return self

    def clean(self) -> "PipelineBuilder":
        self._ensure_loaded()

        self.df = self.dataframe_utils.clean(self.df)  # type: ignore
        return self

    def with_balancer(self, balancer: Type[BalancerProcol]) -> "PipelineBuilder":
        self._ensure_loaded()
        self.df = balancer(self.df).apply()  # type: ignore
        return self

    def with_features(self, features: list[Type[FeatureProcol]]) -> "PipelineBuilder":
        self._ensure_loaded()
        for feature in features:
            self.df = feature(self.df).apply()  # type: ignore
        return self

    def build(self) -> Pipeline:
        self._ensure_loaded()

        return Pipeline(self.df)  # type: ignore

    def _ensure_loaded(self):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Use load() method first.")

    @classmethod
    def with_dataset(cls, dataset_path: str):
        features = [TimeHoursFeature, HourOfDayFeature, TimeSincePreviousFeature]
        return (
            PipelineBuilder(DatasetLoader, DatasetCleaner)
            .load(dataset_path)
            .clean()
            # .with_balancer(OversamplingBalancer)
            .with_features(features)
            .build()
        )

    @classmethod
    def default(cls):
        dataset_path = str(settings.data_path / "creditcard.csv.zip")
        return PipelineBuilder.with_dataset(dataset_path)
