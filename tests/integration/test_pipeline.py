import polars as pl
import pandas as pd
import numpy as np
from features.amount_bin_feature import AmountBinFeature
from features.amount_log_feature import AmountLogFeature
from features.hour_of_day_feature import HourOfDayFeature
from features.is_night_feature import IsNightFeature
from features.time_hours_feature import TimeHoursFeature
from features.time_since_previous_feature import TimeSincePreviousFeature
from dataset_pipeline import DatasetPipeline, DatasetPipelineBuilder
from config import settings
from factory import (
    create_data_pipeline_from_path_without_balancer,
    create_data_pipeline_from_path_with_oversampling_balancer,
)


def test_pipeline_builder():
    current = create_data_pipeline_from_path_without_balancer(
        str(settings.fixtures_path / "fake_dataset.csv.zip")
    )

    assert isinstance(current.df, pl.DataFrame)
    assert current.df.shape == (10, 38)
    assert set(current.df.columns) == {
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
        "Class",
        "TimeHours",
        "HourOfDay",
        "TxnTimeSec",
        "TimeSincePrevSec",
        "isNight",
        "amountLog",
        "amountBin",
    }


def test_data_pipeline_split():

    pipeline = create_data_pipeline_from_path_with_oversampling_balancer(
        str(settings.fixtures_path / "fake_dataset.csv.zip")
    )
    assert isinstance(pipeline, DatasetPipeline)

    test_size = 0.20
    X_tr, X_te, y_tr, y_te = pipeline.split(test_size=test_size, use_cache=False)

    assert isinstance(X_tr, pd.DataFrame)
    assert isinstance(X_te, pd.DataFrame)
    assert isinstance(y_tr, pd.Series)
    assert isinstance(y_te, pd.Series)

    total_rows = len(pipeline.df)
    expected_test = int(np.ceil(total_rows * test_size))
    assert len(X_te) == expected_test
    assert len(y_te) == expected_test

    counts = y_tr.value_counts().to_dict()
    assert counts[0] == counts[1]

    assert list(X_tr.columns) == list(X_te.columns)
    assert "amountBin" not in X_tr.columns
    assert X_tr.select_dtypes(include="object").empty
    assert X_te.select_dtypes(include="object").empty


def test_split_is_deterministic():
    pipeline = create_data_pipeline_from_path_with_oversampling_balancer(
        str(settings.fixtures_path / "fake_dataset.csv.zip")
    )
    assert isinstance(pipeline, DatasetPipeline)

    test_size = 0.2
    X_train, X_test, y_train, y_test = pipeline.split(test_size=test_size)

    X_train_2, X_test_2, y_train_2, y_test_2 = pipeline.split(test_size=test_size)

    assert X_train.equals(X_train_2)
    assert X_test.equals(X_test_2)
    assert y_train.equals(y_train_2)
    assert y_test.equals(y_test_2)


def test_get_feature_list():
    assert DatasetPipelineBuilder.get_standards_features() == [
        TimeHoursFeature,
        HourOfDayFeature,
        TimeSincePreviousFeature,
        IsNightFeature,
        AmountLogFeature,
        AmountBinFeature,
    ]
