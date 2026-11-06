"""Load the experiment configuration from YAML."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "experiment.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())
