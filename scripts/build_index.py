#!/usr/bin/env python3
"""Build only the FAISS index for a given config, reusing existing embeddings."""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from retrieval.indexing import VectorIndex
from utils import config as config_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from existing embeddings")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--embeddings",
        help="Path to .npy embeddings file (default: data/processed/<exp_name>_embeddings.npy, "
             "falling back to baseline_embeddings.npy)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment_name"]
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    # Resolve embeddings file
    if args.embeddings:
        embeddings_file = Path(args.embeddings)
    else:
        candidate = processed_dir / f"{exp_name}_embeddings.npy"
        baseline = processed_dir / "baseline_embeddings.npy"
        embeddings_file = candidate if candidate.exists() else baseline

    if not embeddings_file.exists():
        logger.error(f"Embeddings not found: {embeddings_file}")
        sys.exit(1)

    index_file = processed_dir / f"{exp_name}_index.faiss"

    logger.info(f"Config:     {args.config}")
    logger.info(f"Embeddings: {embeddings_file}")
    logger.info(f"Output:     {index_file}")

    embeddings = np.load(str(embeddings_file))
    logger.info(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    index_config = config_utils.get_index_config(config)
    vector_index = VectorIndex(index_config=index_config, embedding_dim=embeddings.shape[1])
    vector_index.build(embeddings, verbose=True)
    vector_index.save(str(index_file))

    logger.info(f"Done: {index_file}")


if __name__ == "__main__":
    main()
