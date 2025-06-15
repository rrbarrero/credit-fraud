import joblib
from typing import Type
import polars as pl
import pandas as pd
from dataclasses import dataclass
from balancers.balancer import BalancerProtocol
from balancers.oversampling_balancer import OversamplingBalancer
from features.amount_bin_feature import AmountBinFeature
from features.amount_log_feature import AmountLogFeature
from features.feature import FeatureProcol
from features.hour_of_day_feature import HourOfDayFeature
from features.is_night_feature import IsNightFeature
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
class DatasetPipeline:
    df: pl.DataFrame
    balancer: Type[BalancerProtocol] = OversamplingBalancer

    @property
    def X(self) -> pl.DataFrame:
        return self.df.drop(["Class"])

    @property
    def y(self) -> pl.Series:
        return self.df["Class"]

    def apply_balancer(self, X_tr: pd.DataFrame, y_tr: pd.Series, X_te):
        return self.balancer().fit_resample(X_tr, y_tr, X_te)

    def split(self, test_size: float = 0.3, use_cache: bool = False):
        if use_cache:
            cache = self._load_cache()
            if cache is not None:
                print("Loading data from cache...")
                return cache

        n_classes = self.y.n_unique()

        if isinstance(test_size, int) and test_size < n_classes:
            raise ValueError(
                f"test_size={test_size} is less than the number of classes ({n_classes}). "
                "Use at least n_classes or a float value ≤1.0."
            )

        X_tr_pd, X_te_pd, y_tr_pd, y_te_pd = train_test_split(
            self.X.to_pandas(),
            self.y.to_pandas(),
            stratify=self.y.to_pandas(),
            test_size=test_size,
            random_state=42,
        )
        X_tr_bal, y_tr_bal, X_te_bal = self.apply_balancer(X_tr_pd, y_tr_pd, X_te_pd)

        for df in (X_tr_bal, X_te_bal):
            df.drop(columns="amountBin", errors="ignore", inplace=True)

        self._save_cache(X_tr_bal, X_te_bal, y_tr_bal, y_te_pd)

        return (
            X_tr_bal,
            X_te_bal,
            y_tr_bal,
            y_te_pd,
        )

    def _save_cache(self, X_tr_bal, X_te_pd, y_tr_bal, y_te_pd):
        joblib.dump(X_tr_bal, settings.cache_path / "X_tr_bal.joblib")
        joblib.dump(X_te_pd, settings.cache_path / "X_te_pd.joblib")
        joblib.dump(y_tr_bal, settings.cache_path / "y_tr_bal.joblib")
        joblib.dump(y_te_pd, settings.cache_path / "y_te_pd.joblib")

    def _load_cache(self):
        try:
            X_tr_bal = joblib.load(settings.cache_path / "X_tr_bal.joblib")
            X_te_pd = joblib.load(settings.cache_path / "X_te_pd.joblib")
            y_tr_bal = joblib.load(settings.cache_path / "y_tr_bal.joblib")
            y_te_pd = joblib.load(settings.cache_path / "y_te_pd.joblib")
        except FileNotFoundError:
            print("Cache not found. Continuing normal flow...")
            return None

        return X_tr_bal, X_te_pd, y_tr_bal, y_te_pd


class DatasetPipelineBuilder:
    def __init__(
        self,
        dataset_loader: DatasetLoaderProtocol,
        dataframe_utils: DatasetCleanerProtocol,
    ):
        self.dataset_loader = dataset_loader
        self.dataframe_utils = dataframe_utils
        self.features: list[Type[FeatureProcol]] = []
        self.dataset_path: str | None = None

    def _load(self) -> pl.DataFrame:
        if self.dataset_path is None:
            raise ValueError("Dataset path is not set. Use with_path() method first.")
        return self.dataset_loader.handle(self.dataset_path)

    def _clean(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.dataframe_utils.clean(df)

    def with_path(self, dataset_path: str) -> "DatasetPipelineBuilder":
        self.dataset_path = dataset_path
        return self

    def with_features(
        self, features: list[Type[FeatureProcol]]
    ) -> "DatasetPipelineBuilder":
        self.features = features  # type: ignore
        return self

    def build(self) -> DatasetPipeline:
        df = self._load()
        df = self._clean(df)

        for feature_class in self.features:
            df = feature_class(df).apply()

        return DatasetPipeline(df)

    @staticmethod
    def get_standards_features():
        return [
            TimeHoursFeature,
            HourOfDayFeature,
            TimeSincePreviousFeature,
            IsNightFeature,
            AmountLogFeature,
            AmountBinFeature,
        ]

    @classmethod
    def default(cls):
        dataset_path = str(settings.data_path / "creditcard.csv.zip")

        return (
            DatasetPipelineBuilder(DatasetLoader, DatasetCleaner)
            .with_path(dataset_path)
            .with_features(cls.get_standards_features())
            .build()
        )
