import polars as pl
import pytest
from polars.testing import assert_frame_equal

from features.time_since_previous_feature import TimeSincePreviousFeature


def test_time_since_previous_feature_simple():
    df_input = pl.DataFrame({"Time": [0, 10, 25, 25, 100]})
    df_out = TimeSincePreviousFeature(df_input).apply()

    df_expected = (
        df_input.lazy()
        .sort("Time")
        .with_columns(
            [
                pl.col("Time")
                .diff(1)
                .fill_null(0)
                .cast(pl.Float64)
                .alias("TimeSincePrevSec"),
                ((pl.col("Time") % 86400).floor().cast(pl.Float64)).alias("TxnTimeSec"),
            ]
        )
        .collect()
    )

    assert_frame_equal(df_out, df_expected, check_dtypes=True, check_exact=True)


@pytest.fixture
def random_df():
    return pl.DataFrame({"Time": [5, 2, 9, 2, 15]})


def test_with_unsorted_times(random_df):
    df_input = random_df.sample(fraction=1.0, shuffle=True, seed=0)
    df_out = TimeSincePreviousFeature(df_input).apply()

    times = df_out["Time"].to_list()
    assert times == sorted(times)

    diffs = df_out["TimeSincePrevSec"].to_list()
    expected_diffs = [0.0] + [curr - prev for prev, curr in zip(times, times[1:])]
    assert diffs == expected_diffs

    txn_secs = df_out["TxnTimeSec"].to_list()
    assert txn_secs == [float(t % 86400) for t in times]
