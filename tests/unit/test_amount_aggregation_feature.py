import polars as pl
from features.amount_aggregation_feature import AmountAggregationFeature


def test_amount_aggregation_feature():
    df = pl.DataFrame(
        {
            "Time": [1000, 1010, 4600],
            "Amount": [10.0, 20.0, 60.0],
        }
    )

    result_df = AmountAggregationFeature(df).apply()

    expected_values = {
        "amount_mean_1h": [15.0, 40.0, 60.0],
        "amount_sum_1h": [30.0, 80, 60.0],
        "amount_to_mean_ratio_1h": [
            0.6666666222222252,
            0.49999998750000035,
            0.9999999833333336,
        ],
    }

    for col, expected in expected_values.items():
        assert result_df[col].to_list() == expected, f"Column {col} did not match"


def test_amount_aggregation_empty_df():
    df = pl.DataFrame(
        {"Time": [], "Amount": []}, schema={"Time": pl.Int64, "Amount": pl.Float64}
    )
    result_df = AmountAggregationFeature(df).apply()
    assert result_df.shape[0] == 0
