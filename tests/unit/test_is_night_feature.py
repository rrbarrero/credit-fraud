import pytest
import polars as pl
from features.is_night_feature import IsNightFeature


@pytest.mark.parametrize(
    "time_seconds, expected",
    [
        (23 * 3600, 1),  # 23h
        (23 * 3600 + 59, 1),
        (0, 1),
        (7 * 3600 + 59, 1),
        (8 * 3600, 0),
        (12 * 3600, 0),
        (22 * 3600 + 30, 0),
    ],
)
def test_is_night_default_bounds(time_seconds, expected):
    df = pl.DataFrame({"Time": [time_seconds]})
    out = IsNightFeature(df).apply()
    assert out["isNight"][0] == expected


@pytest.mark.parametrize(
    "night_start, night_end, time_seconds, expected",
    [
        (22, 6, 22 * 3600, 1),  # 22h
        (22, 6, 5 * 3600 + 1, 1),
        (22, 6, 6 * 3600, 0),
        (22, 6, 21 * 3600, 0),
    ],
)
def test_is_night_custom_bounds(night_start, night_end, time_seconds, expected):
    df = pl.DataFrame({"Time": [time_seconds]})
    feature = IsNightFeature(df, night_start=night_start, night_end=night_end)
    out = feature.apply()
    assert out["isNight"][0] == expected
