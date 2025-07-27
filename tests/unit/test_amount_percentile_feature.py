import polars as pl
from features.amount_percentile_feature import AmountPercentileFeature
from polars.testing import assert_frame_equal


def test_amount_percentile_feature_simple():
    df = pl.DataFrame({"Amount": [10.0, 20.0, 30.0, 40.0, 50.0]})
    expected_df = pl.DataFrame(
        {
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "amount_percentile": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )
    result_df = AmountPercentileFeature(df).apply()
    assert_frame_equal(result_df, expected_df)


def test_amount_percentile_feature_with_duplicates():
    df = pl.DataFrame({"Amount": [10.0, 20.0, 20.0, 30.0]})
    expected_df = pl.DataFrame(
        {
            "Amount": [10.0, 20.0, 20.0, 30.0],
            "amount_percentile": [0.0, 0.5, 0.5, 1.0],
        }
    )
    result_df = AmountPercentileFeature(df).apply()
    assert_frame_equal(result_df, expected_df)


def test_amount_percentile_feature_empty_df():
    df = pl.DataFrame({"Amount": pl.Series(dtype=pl.Float64)})
    expected_df = pl.DataFrame(
        {
            "Amount": pl.Series(dtype=pl.Float64),
            "amount_percentile": pl.Series(dtype=pl.Float64),
        }
    )
    result_df = AmountPercentileFeature(df).apply()
    assert_frame_equal(result_df, expected_df)


def test_amount_percentile_feature_single_value():
    df = pl.DataFrame({"Amount": [100.0]})
    expected_df = pl.DataFrame(
        {
            "Amount": [100.0],
            "amount_percentile": [0.0],
        }
    )
    result_df = AmountPercentileFeature(df).apply()
    assert_frame_equal(result_df, expected_df)


def test_amount_percentile_feature_unsorted_input():
    df = pl.DataFrame({"Amount": [50.0, 10.0, 30.0, 20.0, 40.0]})
    expected_df = pl.DataFrame(
        {
            "Amount": [50.0, 10.0, 30.0, 20.0, 40.0],
            "amount_percentile": [1.0, 0.0, 0.5, 0.25, 0.75],
        }
    )
    result_df = AmountPercentileFeature(df).apply()
    assert_frame_equal(result_df, expected_df)


def test_amount_percentile_feature_with_other_columns():
    df = pl.DataFrame({"Time": [1, 2, 3], "Amount": [10.0, 20.0, 30.0]})
    expected_df = pl.DataFrame(
        {
            "Time": [1, 2, 3],
            "Amount": [10.0, 20.0, 30.0],
            "amount_percentile": [0.0, 0.5, 1.0],
        }
    )
    result_df = AmountPercentileFeature(df).apply()
    assert_frame_equal(result_df, expected_df)
