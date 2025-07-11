import pytest
from utils.filesystem_utils import DatasetLoader
from config import settings


@pytest.mark.acceptance
def test_load_dataset():
    current = DatasetLoader.handle(str(settings.data_path / "creditcard.csv.zip"))

    assert current.shape == (284807, 31)
