import random
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from features.time_hours_feature import TimeHoursFeature


def test_time_hours_feature_simple():
    df_input = pl.DataFrame({"Time": [0, 3600, 4500, 86399, 86400, 90000]})
    df_out = TimeHoursFeature(df_input).apply()

    df_expected = df_input.with_columns(
        ((pl.col("Time") / 3600) % 24).alias("TimeHours")
    )
    df_expected = df_expected.with_columns(pl.col("TimeHours").cast(pl.Float64))

    assert_frame_equal(df_out, df_expected, check_dtypes=True, check_exact=True)


@pytest.mark.parametrize(
    "time_val, expected",
    [
        (0, 0.0),
        (3600, 1.0),
        (4500, 1.25),
        (86399, pytest.approx(23.99972222)),
        (86400, 0.0),
        (90000, 1.0),
    ],
)
def test_time_hours_values(time_val, expected):
    df_input = pl.DataFrame({"Time": [time_val]})
    df_out = TimeHoursFeature(df_input).apply()
    value = df_out["TimeHours"][0]
    assert value == expected


def test_with_random_times():
    random.seed(0)

    times = [random.randint(0, 1000000) for _ in range(100)]
    df_input = pl.DataFrame({"Time": times})
    df_out = TimeHoursFeature(df_input).apply()

    tol = 1e-12

    for t, hours in zip(times, df_out["TimeHours"].to_list()):
        expected = (t / 3600) % 24
        assert abs(hours - expected) <= tol
