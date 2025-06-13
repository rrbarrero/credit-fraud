import polars as pl
from pipeline import PipelineBuilder
from config import settings


def test_pipeline():
    current = PipelineBuilder.with_dataset(
        str(settings.fixtures_path / "fake_dataset.csv.zip")
    )

    assert isinstance(current.df, pl.DataFrame)
    assert current.df.shape == (10, 31)
