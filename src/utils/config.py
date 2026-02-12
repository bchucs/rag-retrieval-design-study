"""Configuration loading utilities

Centralized config loading to avoid hardcoding values across scripts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


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


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two config dictionaries (override_config takes precedence)

    Args:
        base_config: Base configuration
        override_config: Override configuration (partial configs allowed)

    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(base_config)

    def deep_merge(base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    deep_merge(merged, override_config)
    return merged


# ===== Factory functions to create config objects from YAML =====

def get_chunk_config(config: Dict[str, Any] = None):
    """Create ChunkConfig from configuration dict

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        ChunkConfig instance
    """
    from data.chunking import ChunkConfig

    if config is None:
        config = load_config()

    chunk_cfg = config['chunking']
    return ChunkConfig(
        chunk_size=chunk_cfg['chunk_size'],
        overlap=chunk_cfg['overlap'],
        encoding_name=chunk_cfg.get('encoding_name', 'cl100k_base')
    )


def get_embedding_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get embedding configuration

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        Embedding config dict with model_name, batch_size, cache_dir
    """
    if config is None:
        config = load_config()
    return config['embedding']


def get_index_config(config: Dict[str, Any] = None):
    """Create IndexConfig from configuration dict

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        IndexConfig instance
    """
    from retrieval.indexing import IndexConfig

    if config is None:
        config = load_config()

    index_cfg = config['indexing']
    return IndexConfig(
        index_type=index_cfg['index_type'],
        n_clusters=index_cfg.get('n_clusters', 100),
        nprobe=index_cfg.get('nprobe', 10)
    )


def get_retrieval_config(config: Dict[str, Any] = None):
    """Create RetrievalConfig from configuration dict

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        RetrievalConfig instance
    """
    from retrieval.retriever import RetrievalConfig

    if config is None:
        config = load_config()

    retr_cfg = config['retrieval']
    return RetrievalConfig(
        top_k=retr_cfg['top_k'],
        distance_threshold=retr_cfg.get('distance_threshold')
    )


def get_generation_config(config: Dict[str, Any] = None):
    """Create GenerationConfig from configuration dict

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        GenerationConfig instance
    """
    from generation.llm import GenerationConfig

    if config is None:
        config = load_config()

    gen_cfg = config['generation']
    return GenerationConfig(
        model_name=gen_cfg['model_name'],
        temperature=gen_cfg.get('temperature', 0.0),
        max_tokens=gen_cfg.get('max_tokens', 500),
        api_key=gen_cfg.get('api_key')
    )


# ===== Legacy helper functions (for backward compatibility) =====

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