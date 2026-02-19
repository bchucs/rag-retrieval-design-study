#!/usr/bin/env python3
"""Build retrieval system artifacts

This script builds the complete retrieval system for a given configuration,
including chunks, embeddings, and vector index.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
import argparse
import logging
import json
import hashlib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from data.loader import ArXivLoader
from data.chunking import TextChunker
from retrieval.embeddings import EmbeddingModel
from retrieval.indexing import VectorIndex
from utils import config as config_utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    path = Path(config_path)
    if not path.exists():
        project_root = Path(__file__).resolve().parent.parent
        alt_path = project_root / "experiments" / "configs" / config_path
        if alt_path.exists():
            path = alt_path
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_artifact_hash(chunk_config, emb_config: Dict, max_docs: Optional[int]) -> str:
    """Compute hash of chunking and embedding configuration

    This determines if artifacts can be shared between experiments.

    Args:
        chunk_config: ChunkConfig object
        emb_config: Embedding config dict
        max_docs: Maximum documents (or None)
    """
    # Create a canonical representation
    config_str = json.dumps({
        'chunk_size': chunk_config.chunk_size,
        'overlap': chunk_config.overlap,
        'encoding_name': chunk_config.encoding_name,
        'embedding_model': emb_config.get('model_name'),
        'max_docs': max_docs
    }, sort_keys=True)

    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def find_reusable_artifacts(
    processed_dir: Path,
    chunk_config,
    emb_config: Dict,
    max_docs: Optional[int]
) -> Optional[Tuple[Path, Path]]:
    """Find existing artifacts that match the current configuration

    Args:
        processed_dir: Directory for processed data
        chunk_config: ChunkConfig object
        emb_config: Embedding config dict
        max_docs: Maximum documents

    Returns:
        Tuple of (chunks_file, embeddings_file) if found, None otherwise
    """
    # Check if baseline artifacts exist with matching config
    baseline_chunks = processed_dir / "baseline_chunks.json"
    baseline_embeddings = processed_dir / "baseline_embeddings.npy"
    baseline_metadata = processed_dir / "baseline_metadata.json"

    if baseline_chunks.exists() and baseline_embeddings.exists() and baseline_metadata.exists():
        # Load baseline metadata to check if configs match
        with open(baseline_metadata, 'r') as f:
            baseline_meta = json.load(f)

        # Check if chunking and embedding configs match
        current_hash = compute_artifact_hash(chunk_config, emb_config, max_docs)
        baseline_hash = baseline_meta.get('artifact_hash')

        if current_hash == baseline_hash:
            logger.info(f"✓ Found matching baseline artifacts (hash: {current_hash})")
            return (baseline_chunks, baseline_embeddings)
        else:
            logger.info(f"✗ Baseline artifacts exist but config differs")
            logger.info(f"  Current hash: {current_hash}, Baseline hash: {baseline_hash}")

    return None


def save_artifact_metadata(
    processed_dir: Path,
    exp_name: str,
    chunk_config,
    emb_config: Dict,
    max_docs: Optional[int],
    num_chunks: int,
    embedding_dim: int
):
    """Save metadata about generated artifacts for future reuse checks

    Args:
        processed_dir: Directory for processed data
        exp_name: Experiment name
        chunk_config: ChunkConfig object
        emb_config: Embedding config dict
        max_docs: Maximum documents
        num_chunks: Number of chunks created
        embedding_dim: Embedding dimension
    """
    metadata = {
        'experiment_name': exp_name,
        'artifact_hash': compute_artifact_hash(chunk_config, emb_config, max_docs),
        'chunk_config': {
            'chunk_size': chunk_config.chunk_size,
            'overlap': chunk_config.overlap,
            'encoding_name': chunk_config.encoding_name
        },
        'embedding_config': {
            'model_name': emb_config.get('model_name'),
            'batch_size': emb_config.get('batch_size')
        },
        'max_docs': max_docs,
        'num_chunks': num_chunks,
        'embedding_dim': embedding_dim
    }

    metadata_file = processed_dir / f"{exp_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved artifact metadata to {metadata_file.name}")


def build_retrieval_system(config: Dict[str, Any], max_docs: int = None, force_rebuild: bool = False):
    """Build retrieval system artifacts for a given configuration

    Args:
        config: Experiment configuration
        max_docs: Maximum number of documents to process (for testing)
        force_rebuild: If True, regenerate artifacts even if reusable ones exist
    """
    exp_name = config['experiment_name']
    logger.info(f"="*60)
    logger.info(f"Building retrieval system: {exp_name}")
    logger.info(f"="*60)

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / config['data']['data_dir']
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # File paths for this experiment's artifacts
    chunks_file = processed_dir / f"{exp_name}_chunks.json"
    embeddings_file = processed_dir / f"{exp_name}_embeddings.npy"
    index_file = processed_dir / f"{exp_name}_index.faiss"

    # Get configs early for artifact reuse check
    chunk_config = config_utils.get_chunk_config(config)
    emb_config = config_utils.get_embedding_config(config)

    # Check if we can reuse existing artifacts
    logger.info("\n[Step 1/5] Checking for reusable artifacts...")
    reusable = find_reusable_artifacts(processed_dir, chunk_config, emb_config, max_docs)
    artifacts_reused = False

    if reusable and exp_name != "baseline" and not force_rebuild:
        # Reuse baseline artifacts
        artifacts_reused = True
        baseline_chunks_file, baseline_embeddings_file = reusable
        logger.info(f"♻️  Reusing baseline artifacts:")
        logger.info(f"  - Chunks: {baseline_chunks_file.name}")
        logger.info(f"  - Embeddings: {baseline_embeddings_file.name}")

        # Load chunks
        with open(baseline_chunks_file, 'r') as f:
            chunks = json.load(f)
        logger.info(f"  Loaded {len(chunks)} chunks")

        # Load embedding model and embeddings
        embedding_model = EmbeddingModel(
            model_name=emb_config['model_name'],
            batch_size=emb_config['batch_size'],
            cache_dir=emb_config.get('cache_dir')
        )
        embedding_model.load()
        embeddings = embedding_model.load_embeddings(str(baseline_embeddings_file))
        logger.info(f"  Loaded embeddings with shape {embeddings.shape}")

        # Create symlinks for this experiment (optional, for clarity)
        if not chunks_file.exists():
            chunks_file.symlink_to(baseline_chunks_file.name)
            logger.info(f"  Created symlink: {chunks_file.name} -> {baseline_chunks_file.name}")
        if not embeddings_file.exists():
            embeddings_file.symlink_to(baseline_embeddings_file.name)
            logger.info(f"  Created symlink: {embeddings_file.name} -> {baseline_embeddings_file.name}")

    else:
        # Generate new artifacts
        logger.info(f"Generating new artifacts for {exp_name}")

        # 1. Load data
        logger.info("\n[Step 1a/5] Loading arXiv CS corpus...")
        loader = ArXivLoader(str(data_dir), max_docs=max_docs)
        documents = loader.load_corpus()
        logger.info(f"Loaded {len(documents)} documents")

        # 2. Chunk documents
        logger.info("\n[Step 2/5] Chunking documents...")
        chunker = TextChunker(chunk_config)
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Save chunks
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved chunks to {chunks_file}")

        # 3. Generate embeddings
        logger.info("\n[Step 3/5] Generating embeddings...")
        embedding_model = EmbeddingModel(
            model_name=emb_config['model_name'],
            batch_size=emb_config['batch_size'],
            cache_dir=emb_config.get('cache_dir')
        )
        embedding_model.load()

        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_model.embed_texts(
            chunk_texts,
            show_progress=True,
            normalize=True
        )
        logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")

        # Save embeddings
        embedding_model.save_embeddings(embeddings, str(embeddings_file))

        # Save metadata for future reuse
        save_artifact_metadata(
            processed_dir, exp_name, chunk_config, emb_config, max_docs,
            len(chunks), embedding_model.embedding_dim
        )

    # 4. Build index
    logger.info("\n[Step 4/5] Building vector index...")
    index_config = config_utils.get_index_config(config)
    vector_index = VectorIndex(
        index_config=index_config,
        embedding_dim=embedding_model.embedding_dim
    )
    vector_index.build(embeddings, verbose=True)

    # Save index
    vector_index.save(str(index_file))

    logger.info("\n" + "="*60)
    logger.info("Retrieval system build complete!")
    logger.info(f"Artifacts location: {processed_dir}")
    if artifacts_reused:
        logger.info(f"  - Chunks: {chunks_file.name} (symlink to baseline)")
        logger.info(f"  - Embeddings: {embeddings_file.name} (symlink to baseline)")
    else:
        logger.info(f"  - Chunks: {chunks_file.name}")
        logger.info(f"  - Embeddings: {embeddings_file.name}")
    logger.info(f"  - Index: {index_file.name}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Build retrieval system artifacts")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/baseline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force regeneration of artifacts even if reusable ones exist"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    build_retrieval_system(config, max_docs=args.max_docs, force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()