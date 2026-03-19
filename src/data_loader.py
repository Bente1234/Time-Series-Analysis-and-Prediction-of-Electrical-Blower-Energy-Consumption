from pathlib import Path
from typing import Dict, List

import pandas as pd


class DataLoader:
    """Loads and combines raw CSV data files."""

    def __init__(self, config: Dict):
        self.config = config
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_dir = self.project_root / self.config["paths"]["data_dir"]
        self.input_files: List[str] = self.config["data"]["input_files"]

    def load_raw_data(self) -> pd.DataFrame:
        """Read all configured CSV files and combine them into one dataframe."""
        dataframes = []

        for file_name in self.input_files:
            file_path = self.data_dir / file_name

            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")

            df = pd.read_csv(file_path)
            df["source_file"] = file_name
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df