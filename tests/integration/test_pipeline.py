import polars as pl
import pandas as pd
from features.amount_log_feature import AmountLogFeature
from features.hour_of_day_feature import HourOfDayFeature
from features.is_night_feature import IsNightFeature
from features.time_hours_feature import TimeHoursFeature
from features.time_since_previous_feature import TimeSincePreviousFeature
from pipeline import DataPipeline, PipelineBuilder
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
    assert current.df.shape == (10, 37)
    assert current.df.columns == [
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
        "TxnTime",
        "TimeSincePrevSec",
        "isNight",
        "amountLog",
    ]


def test_data_pipeline_split():
    pipeline = create_data_pipeline_from_path_with_oversampling_balancer(
        str(settings.fixtures_path / "fake_dataset.csv.zip")
    )
    assert isinstance(pipeline, DataPipeline)

    test_size = 0.5
    X_train, X_test, y_train, y_test = pipeline.split(test_size=test_size)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    total_rows = pipeline.df.shape[0]
    expected_train = int(total_rows * (1 - test_size))
    expected_test = total_rows - expected_train

    n_features = pipeline.df.shape[1] - 1

    assert X_train.shape == (expected_train, n_features)
    assert X_test.shape == (expected_test, n_features)
    assert y_train.shape == (expected_train,)
    assert y_test.shape == (expected_test,)


def test_split_is_deterministic():
    pipeline = create_data_pipeline_from_path_with_oversampling_balancer(
        str(settings.fixtures_path / "fake_dataset.csv.zip")
    )
    assert isinstance(pipeline, DataPipeline)

    test_size = 0.1
    X_train, X_test, y_train, y_test = pipeline.split(test_size=test_size)

    X_train_2, X_test_2, y_train_2, y_test_2 = pipeline.split(test_size=test_size)

    assert X_train.equals(X_train_2)
    assert X_test.equals(X_test_2)
    assert y_train.equals(y_train_2)
    assert y_test.equals(y_test_2)


def test_get_feature_list():
    assert PipelineBuilder.get_standards_features() == [
        TimeHoursFeature,
        HourOfDayFeature,
        TimeSincePreviousFeature,
        IsNightFeature,
        AmountLogFeature,
    ]
