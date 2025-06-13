import polars as pl
import pytest
from polars.testing import assert_frame_equal

from features.time_since_previous_feature import TimeSincePreviousFeature


def test_time_since_previous_feature_simple():
    df_input = pl.DataFrame({"Time": [0, 10, 25, 25, 100]})
    df_out = TimeSincePreviousFeature(df_input).apply()

    df_expected = (
        df_input.lazy()
        .with_columns(pl.duration(seconds=pl.col("Time")).alias("TxnTime"))
        .sort("TxnTime")
        .with_columns(
            pl.col("TxnTime")
            .diff()
            .dt.total_seconds()
            .fill_null(0)
            .alias("TimeSincePrevSec")
        )
        .collect()
    )

    assert_frame_equal(df_out, df_expected, check_dtypes=True, check_exact=True)


@pytest.fixture
def random_df():
    return pl.DataFrame({"Time": [5, 2, 9, 2, 15]})


def test_with_unsorted_times(random_df):
    df_input = random_df.sample(fraction=1.0, shuffle=True, seed=0)

    transformer = TimeSincePreviousFeature(df_input)
    df_out = transformer.apply()

    times = df_out["Time"].to_list()
    assert times == sorted(times)

    diffs = df_out["TimeSincePrevSec"].to_list()
    expected_diffs = [0.0] + [curr - prev for prev, curr in zip(times, times[1:])]
    assert diffs == expected_diffs
