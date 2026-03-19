from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Loads project configuration from a YAML file."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)

    def load(self) -> Dict[str, Any]:
        """Read and return the YAML configuration as a dictionary."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a top-level dictionary.")

        return config