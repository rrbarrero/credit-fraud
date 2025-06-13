import pytest
import polars as pl
from utils.data_utils import CreditFraudDataframeUtils


def test_clean_dataframe():
    df = pl.DataFrame(
        {"A": [1, 2, None, 4, 5], "B": [None, 2, 3, 4, None], "C": [1, 2, 3, 4, 5]}
    )

    current = CreditFraudDataframeUtils.clean(df)

    expected = [{"A": 2, "B": 2, "C": 2}, {"A": 4, "B": 4, "C": 4}]

    current_dicts = [dict(row) for row in current.to_dicts()]
    expected_dicts = [dict(row) for row in expected]

    assert sorted(current_dicts, key=lambda x: (x["A"], x["B"], x["C"])) == sorted(
        expected_dicts, key=lambda x: (x["A"], x["B"], x["C"])
    )


def test_clean_dataframe_no_nulls():
    df = pl.DataFrame(
        {"A": [1, 2, None, 4, 5], "B": [None, 2, 3, 4, None], "C": [1, 2, 3, 4, 5]}
    )

    cleaned = CreditFraudDataframeUtils.clean(df)

    assert cleaned.null_count().to_series().sum() == 0


@pytest.fixture
def sample_input_df():
    data = {
        "Time": [1000, 2000, 3000, 4000, 5000, 6000],
        "V1": [0.1, 0.2, 0.1, 0.3, 0.2, 0.1],
        "V2": [1.1, 1.2, 1.1, 1.3, 1.2, 1.1],
        "Amount": [50, 60, 70, 80, 90, 100],
        "Class": [0, 0, 0, 0, 1, 1],
    }
    return pl.DataFrame(data)


def test_balance_credit_fraud_df_structure(sample_input_df):
    result = CreditFraudDataframeUtils.balance(sample_input_df)

    assert isinstance(result, pl.DataFrame)

    assert "Class" in result.columns

    assert result.null_count().to_series().sum() == 0


def test_balance_credit_fraud_df_class_balance(sample_input_df):
    result = CreditFraudDataframeUtils.balance(sample_input_df)

    counts_df = result.select(pl.col("Class").value_counts()).unnest("Class")

    class_counts = {
        row["Class"]: row["count"] for row in counts_df.iter_rows(named=True)
    }

    assert set(class_counts.keys()) == {0, 1}

    assert class_counts[0] == class_counts[1]


def test_balance_credit_fraud_df_column_consistency(sample_input_df):
    result = CreditFraudDataframeUtils.balance(sample_input_df)

    assert set(result.columns) == set(sample_input_df.columns)
