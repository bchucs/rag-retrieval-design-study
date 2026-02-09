#!/usr/bin/env python3
"""Interactive retrieval testing script

Run this after completing the baseline experiment to test retrieval with custom queries.
"""

import sys
sys.path.insert(0, "src")

import json
import numpy as np
from pathlib import Path
from retrieval.embeddings import EmbeddingModel
from retrieval.indexing import VectorIndex, IndexConfig
from retrieval.retriever import Retriever, RetrievalConfig
from utils.config import load_config, get_embedding_model_name, get_retrieval_config


def main():
    print("="*60)
    print("Interactive Retrieval Testing")
    print("="*60)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    embedding_model_name = get_embedding_model_name(config)
    retrieval_cfg = get_retrieval_config(config)
    print(f"✓ Config loaded: {config['experiment_name']}")

    # Load artifacts
    processed_dir = Path("data/processed")

    print("\nLoading artifacts...")

    chunks_file = processed_dir / "baseline_chunks.json"
    index_file = processed_dir / "baseline_index.faiss"

    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found")
        print("Please run: python experiments/scripts/run_baseline.py --max-docs 100")
        return

    with open(chunks_file) as f:
        chunks = json.load(f)
    print(f"✓ Loaded {len(chunks)} chunks")

    # Load embedding model
    print(f"Loading embedding model: {embedding_model_name}...")
    embedding_model = EmbeddingModel(embedding_model_name)
    embedding_model.load()
    print(f"✓ Model loaded (dim={embedding_model.embedding_dim})")

    # Load index
    print("Loading vector index...")
    index_config = IndexConfig(index_type="flat")
    vector_index = VectorIndex(index_config, embedding_model.embedding_dim)
    vector_index.load(str(index_file))
    print(f"✓ Index loaded ({vector_index.n_vectors} vectors)")

    # Create retriever
    retrieval_config = RetrievalConfig(
        top_k=retrieval_cfg['top_k'],
        distance_threshold=retrieval_cfg.get('distance_threshold')
    )
    retriever = Retriever(embedding_model, vector_index, retrieval_config, chunks)

    print("\n" + "="*60)
    print("Ready! Enter queries (or 'quit' to exit)")
    print("="*60 + "\n")

    # Interactive loop
    while True:
        query = input("\nQuery: ").strip()

        if not query or query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        try:
            result = retriever.retrieve(query)

            print(f"\n{'─'*60}")
            print(f"Retrieved {result.num_results} results in {result.latency_ms:.2f}ms")
            print(f"{'─'*60}")

            for i, (doc, score) in enumerate(zip(result.documents, result.distances)):
                print(f"\n{i+1}. Similarity Score: {score:.3f}")
                print(f"   Document ID: {doc.get('doc_id', 'N/A')}")
                print(f"   Title: {doc.get('original_doc', {}).get('title', 'N/A')}")
                print(f"   Chunk: {doc['text'][:200]}...")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
