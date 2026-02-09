"""Configuration loading utilities

Centralized config loading to avoid hardcoding values across scripts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file

    Args:
        config_path: Path to config file. If None, loads baseline config.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to baseline config
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "experiments" / "configs" / "baseline.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_embedding_model_name(config: Dict[str, Any] = None) -> str:
    """Get embedding model name from config

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        Embedding model name
    """
    if config is None:
        config = load_config()
    return config['embedding']['model_name']


def get_generation_model_name(config: Dict[str, Any] = None) -> str:
    """Get generation model name from config

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        Generation model name
    """
    if config is None:
        config = load_config()
    return config['generation']['model_name']


def get_retrieval_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get retrieval configuration

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        Retrieval config dict
    """
    if config is None:
        config = load_config()
    return config['retrieval']


def get_generation_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get generation configuration

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        Generation config dict
    """
    if config is None:
        config = load_config()
    return config['generation']