import polars as pl
import pandas as pd
from pipeline import DataPipeline, PipelineBuilder
from config import settings
from balancers.oversampling_balancer import OversamplingBalancer


def test_pipeline_builder():
    current = PipelineBuilder.with_dataset(
        str(settings.fixtures_path / "fake_dataset.csv.zip")
    ).build()

    assert isinstance(current.df, pl.DataFrame)
    assert current.df.shape == (10, 35)
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
    ]


def test_data_pipeline_split():
    pipeline = (
        PipelineBuilder.with_dataset(
            str(settings.fixtures_path / "fake_dataset.csv.zip")
        )
        .with_balancer(OversamplingBalancer)
        .build()
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
