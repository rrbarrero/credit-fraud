import pytest
import polars as pl
from features.amount_bin_feature import AmountBinFeature


@pytest.mark.parametrize(
    "amount, expected",
    [
        (0, "<10"),
        (9.99, "<10"),
        (10, "10-50"),
        (49.99, "10-50"),
        (50, "50-100"),
        (99.99, "50-100"),
        (100, "100-500"),
        (499.99, "100-500"),
        (500, "500-1000"),
        (999.99, "500-1000"),
        (1000, ">=1000"),
        (1_000_000, ">=1000"),
    ],
)
def test_amount_bin_default(amount, expected):
    df = pl.DataFrame({"Amount": [amount]})
    out = AmountBinFeature(df).apply()
    assert "amountBin" in out.columns
    assert out["amountBin"][0] == expected


def test_amount_bin_multiple_values():
    amounts = [0, 10, 50, 500, 1000, 2000]
    df = pl.DataFrame({"Amount": amounts})
    out = AmountBinFeature(df).apply()
    expected = ["<10", "10-50", "50-100", "500-1000", ">=1000", ">=1000"]
    assert out["amountBin"].to_list() == expected


def test_amount_bin_dtype():
    df = pl.DataFrame({"Amount": [5]})
    out = AmountBinFeature(df).apply()

    assert out.schema["amountBin"] == pl.Utf8


@pytest.mark.parametrize(
    "bins, labels, amounts, expected",
    [
        (
            [5, 15],
            ["low", "mid", "high"],
            [0, 5, 10, 15, 20],
            ["low", "mid", "mid", "high", "high"],
        ),
        ([100], ["small", "large"], [50, 100, 200], ["small", "large", "large"]),
    ],
)
def test_amount_bin_custom(bins, labels, amounts, expected):
    df = pl.DataFrame({"Amount": amounts})
    feature = AmountBinFeature(df, bins=bins, labels=labels)
    out = feature.apply()
    assert out["amountBin"].to_list() == expected


def test_amount_bin_label_mismatch():

    with pytest.raises(ValueError):
        AmountBinFeature(
            pl.DataFrame({"Amount": [1, 2, 3]}),
            bins=[10, 20],
            labels=["a", "b"],
        )
