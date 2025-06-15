import math
import pytest
import polars as pl
from features.amount_log_feature import AmountLogFeature


@pytest.mark.parametrize(
    "amount, expected",
    [
        (0, 0.0),
        (9, math.log(10)),
        (99.5, math.log(100.5)),
        (1_000_000, math.log(1_000_001)),
    ],
)
def test_amount_log_single(amount, expected):
    df = pl.DataFrame({"Amount": [amount]})
    out = AmountLogFeature(df).apply()

    assert "amountLog" in out.columns
    assert "Amount" in out.columns

    val = out["amountLog"].to_list()[0]
    assert val == pytest.approx(expected, rel=1e-6)


def test_amount_log_multiple():
    amounts = [0, 1, 2, 10]
    df = pl.DataFrame({"Amount": amounts})
    out = AmountLogFeature(df).apply()

    assert out.columns == ["Amount", "amountLog"]

    computed = out["amountLog"].to_list()
    expected = [math.log(a + 1) for a in amounts]
    assert computed == pytest.approx(expected, rel=1e-6)


def test_amount_log_dtype():
    df = pl.DataFrame({"Amount": [0, 5]})
    out = AmountLogFeature(df).apply()

    assert out.schema["amountLog"] == pl.Float64
