#!/usr/bin/env python3
"""Run batch experiments with different configurations

This script allows you to run multiple experiments by merging experiment-specific
configs with a base config. Useful for systematic experimentation.

Example:
    # Run all chunking experiments
    python run_experiment_batch.py --base baseline.yaml --experiments experiment1_chunking.yaml

    # Run with limited documents for testing
    python run_experiment_batch.py --base baseline.yaml --experiments experiment1_chunking.yaml --max-docs 100
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
import logging
from typing import List
from pathlib import Path
import yaml

from utils import config as config_utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiment_configs(experiment_file: str, base_config: dict) -> List[dict]:
    """Load experiment configurations from a file

    Args:
        experiment_file: Path to experiment config file
        base_config: Base configuration to merge with

    Returns:
        List of merged configurations, one per experiment
    """
    with open(experiment_file, 'r') as f:
        exp_data = yaml.safe_load(f)

    if 'experiments' not in exp_data:
        raise ValueError(f"Experiment file {experiment_file} must contain 'experiments' key")

    configs = []
    for exp in exp_data['experiments']:
        # Create a config dict with experiment-specific overrides
        exp_config = {
            'experiment_name': exp['name']
        }

        # Merge experiment overrides
        for key in ['chunking', 'indexing', 'retrieval', 'generation']:
            if key in exp:
                exp_config[key] = exp[key]

        # Merge with base config
        merged = config_utils.merge_configs(base_config, exp_config)
        configs.append(merged)

    return configs


def run_single_experiment(config: dict, max_docs: int = None):
    """Run a single experiment with given config

    Args:
        config: Experiment configuration
        max_docs: Maximum number of documents (for testing)
    """
    # Import here to avoid circular dependencies
    from experiments.scripts.build_retrieval_system import build_retrieval_system

    logger.info("\n" + "=" * 80)
    logger.info(f"Starting experiment: {config['experiment_name']}")
    logger.info("=" * 80)

    # Print config summary
    logger.info("\nConfiguration:")
    logger.info(f"  Chunking: size={config['chunking']['chunk_size']}, overlap={config['chunking']['overlap']}")
    logger.info(f"  Indexing: type={config['indexing']['index_type']}")
    logger.info(f"  Retrieval: top_k={config['retrieval']['top_k']}")
    logger.info(f"  Generation: model={config['generation']['model_name']}")

    # Run the experiment
    build_retrieval_system(config, max_docs=max_docs)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--base",
        type=str,
        default="experiments/configs/baseline.yaml",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        required=True,
        help="Path to experiment variations file"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)"
    )
    parser.add_argument(
        "--skip-to",
        type=str,
        default=None,
        help="Skip to specific experiment by name (useful for resuming)"
    )

    args = parser.parse_args()

    # Load base config
    logger.info(f"Loading base config from: {args.base}")
    base_config = config_utils.load_config(args.base)

    # Load experiment configs
    logger.info(f"Loading experiment variations from: {args.experiments}")
    experiment_configs = load_experiment_configs(args.experiments, base_config)
    logger.info(f"Found {len(experiment_configs)} experiment variations")

    # Run experiments
    skipping = args.skip_to is not None

    for i, exp_config in enumerate(experiment_configs):
        exp_name = exp_config['experiment_name']

        # Handle skip-to
        if skipping:
            if exp_name == args.skip_to:
                logger.info(f"Found skip-to experiment: {exp_name}")
                skipping = False
            else:
                logger.info(f"Skipping experiment {i+1}/{len(experiment_configs)}: {exp_name}")
                continue

        logger.info(f"\n\nRunning experiment {i+1}/{len(experiment_configs)}")

        try:
            run_single_experiment(exp_config, max_docs=args.max_docs)
        except KeyboardInterrupt:
            logger.warning("\n\nExperiment interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed with error: {e}")
            logger.error("Continuing to next experiment...")
            continue

    logger.info("\n" + "=" * 80)
    logger.info("All experiments complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
