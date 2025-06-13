from pathlib import PosixPath

from config import settings


def test_config_class():
    assert isinstance(settings.data_path, PosixPath)
    assert "dataset" in str(settings.data_path)
