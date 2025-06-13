from utils.filesystem_utils import load_dataset
from config import settings


def test_load_dataset():
    current = load_dataset(str(settings.data_path / "creditcard.csv.zip"))

    assert current.shape == (284807, 31)
