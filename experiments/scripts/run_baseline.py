#!/usr/bin/env python3
"""Run baseline RAG experiment

This script runs the baseline configuration and establishes initial metrics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import yaml
import argparse
import logging
import json
from typing import Dict, Any
from pathlib import Path

from data.loader import ArXivLoader
from data.chunking import TextChunker, ChunkConfig
from retrieval.embeddings import EmbeddingModel
from retrieval.indexing import VectorIndex, IndexConfig
from retrieval.retriever import Retriever, RetrievalConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_baseline(config: Dict[str, Any], max_docs: int = None):
    """Run baseline experiment

    Args:
        config: Experiment configuration
        max_docs: Maximum number of documents to process (for testing)
    """
    exp_name = config['experiment_name']
    logger.info(f"="*60)
    logger.info(f"Running baseline experiment: {exp_name}")
    logger.info(f"="*60)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / config['data']['data_dir']
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # File paths for caching
    chunks_file = processed_dir / f"{exp_name}_chunks.json"
    embeddings_file = processed_dir / f"{exp_name}_embeddings.npy"
    index_file = processed_dir / f"{exp_name}_index.faiss"

    # 1. Load data
    logger.info("\n[Step 1/5] Loading arXiv CS corpus...")
    loader = ArXivLoader(str(data_dir), max_docs=max_docs)
    documents = loader.load_corpus()
    logger.info(f"Loaded {len(documents)} documents")

    # 2. Chunk documents
    logger.info("\n[Step 2/5] Chunking documents...")
    chunk_config = ChunkConfig(
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['overlap']
    )
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
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size']
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

    # 4. Build index
    logger.info("\n[Step 4/5] Building vector index...")
    index_config = IndexConfig(
        index_type=config['indexing']['index_type'],
        n_clusters=config['indexing'].get('n_clusters', 100)
    )
    vector_index = VectorIndex(
        config=index_config,
        embedding_dim=embedding_model.embedding_dim
    )
    vector_index.build(embeddings, verbose=True)

    # Save index
    vector_index.save(str(index_file))

    # 5. Test retrieval with sample queries
    logger.info("\n[Step 5/5] Testing retrieval with sample queries...")
    retrieval_config = RetrievalConfig(
        top_k=config['retrieval']['top_k'],
        distance_threshold=config['retrieval'].get('distance_threshold')
    )
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_index=vector_index,
        config=retrieval_config,
        chunks=chunks
    )

    # Sample queries
    sample_queries = [
        "deep learning neural networks",
        "natural language processing transformers",
        "computer vision image classification",
    ]

    logger.info("\nSample Retrieval Results:")
    logger.info("-" * 60)

    for query in sample_queries:
        result = retriever.retrieve(query)
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"Retrieved {result.num_results} documents in {result.latency_ms:.2f}ms")
        logger.info(f"Top 3 results:")

        for i, (doc, dist) in enumerate(zip(result.documents[:3], result.distances[:3])):
            title = doc.get('original_doc', {}).get('title', 'N/A')
            logger.info(f"  {i+1}. [score={dist:.3f}] {title[:80]}...")

    logger.info("\n" + "="*60)
    logger.info("Baseline experiment complete!")
    logger.info(f"Artifacts saved to: {processed_dir}")
    logger.info(f"  - Chunks: {chunks_file.name}")
    logger.info(f"  - Embeddings: {embeddings_file.name}")
    logger.info(f"  - Index: {index_file.name}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run baseline RAG experiment")
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
    args = parser.parse_args()

    config = load_config(args.config)
    run_baseline(config, max_docs=args.max_docs)


if __name__ == "__main__":
    main()