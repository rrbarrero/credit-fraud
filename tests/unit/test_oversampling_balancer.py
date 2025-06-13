import pytest
import polars as pl

from balancers.oversampling_balancer import OversamplingBalancer


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
    result = OversamplingBalancer(sample_input_df).apply()

    assert isinstance(result, pl.DataFrame)

    assert "Class" in result.columns

    assert result.null_count().to_series().sum() == 0


def test_balance_credit_fraud_df_class_balance(sample_input_df):
    result = OversamplingBalancer(sample_input_df).apply()

    counts_df = result.select(pl.col("Class").value_counts()).unnest("Class")

    class_counts = {
        row["Class"]: row["count"] for row in counts_df.iter_rows(named=True)
    }

    assert set(class_counts.keys()) == {0, 1}

    assert class_counts[0] == class_counts[1]


def test_balance_credit_fraud_df_column_consistency(sample_input_df):
    result = OversamplingBalancer(sample_input_df).apply()

    assert set(result.columns) == set(sample_input_df.columns)
