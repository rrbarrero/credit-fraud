import polars as pl
from polars.testing import assert_frame_equal
from features.transaction_frequency_feature import TransactionFrequencyFeature


def test_transaction_frequency_feature():
    df = pl.DataFrame(
        {
            "Time": [1000, 1010, 2800, 4540, 8200, 26200],
            "Amount": [10, 20, 5, 50, 100, 200],
        }
    )

    result_df = TransactionFrequencyFeature(df).apply()

    expected_df = pl.DataFrame(
        {
            "Time": [1000, 1010, 2800, 4540, 8200, 26200],
            "Amount": [10, 20, 5, 50, 100, 200],
            "transactions_last_1h": [4, 3, 2, 1, 1, 1],
            "transactions_last_6h": [5, 4, 3, 2, 2, 1],
            "transactions_last_24h": [6, 5, 4, 3, 2, 1],
        }
    ).with_columns(pl.col("Time", "Amount").cast(pl.UInt32))

    assert_frame_equal(result_df, expected_df, check_dtypes=False)


def test_transaction_frequency_empty_df():
    df = pl.DataFrame(
        {"Time": [], "Amount": []}, schema={"Time": pl.Int64, "Amount": pl.Float64}
    )
    result_df = TransactionFrequencyFeature(df).apply()
    assert result_df.shape[0] == 0
