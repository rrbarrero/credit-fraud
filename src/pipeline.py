from typing import Type
import polars as pl
from dataclasses import dataclass
from features.feature import FeatureProcol
from features.hour_of_day_feature import HourOfDayFeature
from features.time_hours_feature import TimeHoursFeature
from utils.data_utils import (
    CreditFraudDataframeUtils,
    CreditFraudDataframeUtilsProtocol,
)
from utils.filesystem_utils import DatasetLoader, DatasetLoaderProtocol
from config import settings


@dataclass
class Pipeline:
    df: pl.DataFrame


class PipelineBuilder:
    def __init__(
        self,
        dataset_loader: DatasetLoaderProtocol,
        dataframe_utils: CreditFraudDataframeUtilsProtocol,
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

    def add_features(self, features: list[Type[FeatureProcol]]) -> "PipelineBuilder":
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
        return (
            PipelineBuilder(DatasetLoader, CreditFraudDataframeUtils)
            .load(dataset_path)
            .clean()
            .add_features([TimeHoursFeature, HourOfDayFeature])
            .build()
        )

    @classmethod
    def default(cls):
        dataset_path = str(settings.data_path / "creditcard.csv.zip")
        return PipelineBuilder.with_dataset(dataset_path)
