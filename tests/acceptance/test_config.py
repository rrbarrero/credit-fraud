from pathlib import PosixPath
from config import settings


def test_config_class():
    assert settings.data_path == PosixPath(
        "/home/roberto/devel/python/fraud-dect/dataset"
    )
