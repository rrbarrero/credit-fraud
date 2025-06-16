from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_path: str
    random_state: int = 42

    @classmethod
    def load(cls):
        current_path = project_path = Path(__file__).parent.parent
        return cls(project_path=str(current_path))

    @property
    def data_path(self):
        return Path(self.project_path) / "dataset"

    @property
    def fixtures_path(self):
        return Path(self.project_path) / "tests" / "__fixtures__"

    @property
    def cache_path(self):
        return Path(self.project_path) / "cache"


settings = Settings.load()
