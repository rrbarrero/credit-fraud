import polars as pl
from utils.data_utils import DatasetCleaner


def test_clean_dataframe():
    df = pl.DataFrame(
        {"A": [1, 2, None, 4, 5], "B": [None, 2, 3, 4, None], "C": [1, 2, 3, 4, 5]}
    )

    current = DatasetCleaner.clean(df)

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

    cleaned = DatasetCleaner.clean(df)

    assert cleaned.null_count().to_series().sum() == 0
