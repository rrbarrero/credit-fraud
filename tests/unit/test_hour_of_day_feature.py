import polars as pl
import pytest
from polars.testing import assert_frame_equal

from features.hour_of_day_feature import HourOfDayFeature


def test_hour_of_day_feature_simple():
    df_input = pl.DataFrame({"Time": [0, 3600, 4500, 86399, 86400, 90000]})
    df_out = HourOfDayFeature(df_input).apply()

    df_expected = df_input.with_columns((pl.col("Time") / 3600).alias("HourOfDay"))

    assert_frame_equal(df_out, df_expected, check_dtypes=True, check_exact=False)


@pytest.mark.parametrize(
    "time_val, expected",
    [
        (0, 0.0),
        (3600, 1.0),
        (4500, 1.25),
        (86399, pytest.approx(23.99972222, rel=1e-5, abs=1e-8)),
        (86400, 24.0),
        (90000, pytest.approx(25.0, rel=1e-5, abs=1e-8)),
    ],
)
def test_individual_values(time_val, expected):
    df_input = pl.DataFrame({"Time": [time_val]})
    df_out = HourOfDayFeature(df_input).apply()
    result = df_out["HourOfDay"][0]
    assert result == expected


def test_random_times():
    import random

    random.seed(0)
    times = [random.randint(0, 1_000_000) for _ in range(200)]
    df_input = pl.DataFrame({"Time": times})
    df_out = HourOfDayFeature(df_input).apply()

    for t, actual in zip(times, df_out["HourOfDay"].to_list()):
        expected = t / 3600
        assert abs(actual - expected) <= 1e-8
