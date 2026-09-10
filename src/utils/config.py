"""Configuration loader for the project."""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file from the project root."""
    root = get_project_root()
    full_path = root / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_intent_taxonomy(taxonomy_path: str = "config/intent_taxonomy.json") -> Dict[str, Any]:
    """Loads the intent taxonomy JSON file."""
    root = get_project_root()
    full_path = root / taxonomy_path
    if not full_path.exists():
        raise FileNotFoundError(f"Taxonomy file not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)
