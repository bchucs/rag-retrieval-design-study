#!/usr/bin/env python3
"""Test end-to-end RAG pipeline

Run this after completing the baseline experiment to test the full RAG system.
"""

import sys
sys.path.insert(0, "src")

import json
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

from retrieval.embeddings import EmbeddingModel
from retrieval.indexing import VectorIndex, IndexConfig
from retrieval.retriever import Retriever, RetrievalConfig
from generation.llm import LLMGenerator, GenerationConfig
from rag_pipeline import RAGPipeline
from utils.config import load_config, get_embedding_model_name, get_retrieval_config, get_generation_config

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("="*60)
    print("End-to-End RAG Pipeline Test")
    print("="*60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        print("Or export OPENAI_API_KEY=your_key_here")
        return

    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    embedding_model_name = get_embedding_model_name(config)
    retrieval_cfg = get_retrieval_config(config)
    generation_cfg = get_generation_config(config)
    print(f"✓ Config loaded: {config['experiment_name']}")
    print(f"  - Embedding model: {embedding_model_name}")
    print(f"  - Generation model: {generation_cfg['model_name']}")
    print(f"  - Top-k: {retrieval_cfg['top_k']}")

    # Load artifacts
    processed_dir = Path("data/processed")

    print("\nLoading baseline artifacts...")

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
    print("✓ Retriever initialized")

    # Initialize LLM generator
    print("Initializing LLM generator...")
    generation_config = GenerationConfig(
        model_name=generation_cfg['model_name'],
        temperature=generation_cfg['temperature'],
        max_tokens=generation_cfg['max_tokens']
    )
    generator = LLMGenerator(generation_config)
    generator.initialize()
    print("✓ LLM generator initialized")

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(retriever, generator)
    print("✓ RAG pipeline ready\n")

    # Sample questions
    sample_questions = [
        "What are the key challenges in training deep neural networks?",
        "How do transformer models use attention mechanisms?",
        "What techniques are used for computer vision tasks?",
    ]

    print("="*60)
    print("Testing RAG Pipeline with Sample Questions")
    print("="*60)

    results = []
    for question in sample_questions:
        try:
            result = rag_pipeline.query(question, log_details=True)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            continue

    # Print summary statistics
    if results:
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)

        stats = rag_pipeline.get_summary_stats(results)
        print(f"Total queries: {stats['num_queries']}")
        print(f"Average total latency: {stats['avg_total_latency_ms']:.2f}ms")
        print(f"  - Retrieval: {stats['avg_retrieval_latency_ms']:.2f}ms")
        print(f"  - Generation: {stats['avg_generation_latency_ms']:.2f}ms")
        print(f"Average documents used: {stats['avg_documents_used']:.1f}")
        print(f"Total tokens used: {stats['total_tokens']}")
        print(f"  - Prompt tokens: {stats['total_prompt_tokens']}")
        print(f"  - Completion tokens: {stats['total_completion_tokens']}")
        print("="*60)

    print("\nRAG pipeline test complete!")


if __name__ == "__main__":
    main()
