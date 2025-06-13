from pathlib import Path
import zipfile
import polars as pl


def load_dataset(file_path: str) -> pl.DataFrame:
    with zipfile.ZipFile(file_path, "r") as z:
        csv_filename = Path(file_path).name.replace(".zip", "")
        with z.open(csv_filename) as f:
            return pl.read_csv(f, schema_overrides={"Time": pl.Float64})
