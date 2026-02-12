#!/usr/bin/env python3
"""Run evaluation on baseline RAG system

This script evaluates the RAG system on the generated question set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import os
import logging
import argparse
from dotenv import load_dotenv

from retrieval.embeddings import EmbeddingModel
from retrieval.indexing import VectorIndex
from retrieval.retriever import Retriever
from generation.llm import LLMGenerator
from rag_pipeline import RAGPipeline
from evaluation.harness import EvaluationHarness
from utils import config as config_utils

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_evaluation(
    config_path: str = None,
    questions_file: str = "data/questions/eval_questions.json",
    chunks_file: str = None,
    index_file: str = None,
    output_file: str = None,
    num_questions: int = None,
    artifacts_name: str = None
):
    """Run evaluation on RAG system

    Args:
        config_path: Path to experiment config YAML file
        questions_file: Path to questions JSON file
        chunks_file: Path to chunks JSON file (auto-derived if None)
        index_file: Path to FAISS index file (auto-derived if None)
        output_file: Path to save results CSV (optional)
        num_questions: Number of questions to evaluate (None for all)
        artifacts_name: Name prefix for artifacts (e.g., 'baseline', 'exp3_index_flat').
                       If None, uses experiment_name from config.
    """
    logger.info("="*60)
    logger.info("RAG System Evaluation")
    logger.info("="*60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Load configuration
    logger.info("\nLoading configuration...")
    if config_path is None:
        config_path = "experiments/configs/baseline.yaml"

    config = config_utils.load_config(config_path)
    exp_name = config['experiment_name']

    emb_config = config_utils.get_embedding_config(config)
    retrieval_config = config_utils.get_retrieval_config(config)
    generation_config = config_utils.get_generation_config(config)
    index_config = config_utils.get_index_config(config)

    # Auto-derive artifact paths if not provided
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"

    # Chunks always use baseline (all experiments share same chunks)
    if chunks_file is None:
        chunks_file = str(processed_dir / "baseline_chunks.json")

    # Index logic:
    # - If --artifacts NOT provided: use baseline_index.faiss
    # - If --artifacts IS provided: use {experiment_name}_index.faiss
    if index_file is None:
        if not artifacts_name:
            index_file = str(processed_dir / "baseline_index.faiss")
        else:
            index_file = str(processed_dir / f"{exp_name}_index.faiss")

    logger.info(f"Config: {exp_name}")
    logger.info(f"  Config file: {config_path}")
    logger.info(f"  Embedding: {emb_config['model_name']}")
    logger.info(f"  Generation: {generation_config.model_name}")
    logger.info(f"  Top-k: {retrieval_config.top_k}")
    logger.info(f"  Distance threshold: {retrieval_config.distance_threshold}")
    logger.info(f"  Chunks: {chunks_file}")
    logger.info(f"  Index: {index_file}")

    # Load artifacts
    logger.info("\nLoading baseline artifacts...")

    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    logger.info(f"✓ Loaded {len(chunks)} chunks")

    # Load embedding model
    logger.info(f"Loading embedding model...")
    embedding_model = EmbeddingModel(
        model_name=emb_config['model_name'],
        batch_size=emb_config['batch_size'],
        cache_dir=emb_config.get('cache_dir')
    )
    embedding_model.load()
    logger.info(f"✓ Model loaded")

    # Load index
    logger.info("Loading vector index...")
    vector_index = VectorIndex(index_config, embedding_model.embedding_dim)
    vector_index.load(str(index_file))
    logger.info(f"✓ Index loaded ({vector_index.n_vectors} vectors)")

    # Create retriever
    retriever = Retriever(embedding_model, vector_index, retrieval_config, chunks)
    logger.info("✓ Retriever initialized")

    # Initialize LLM generator
    logger.info("Initializing LLM generator...")
    generator = LLMGenerator(generation_config)
    generator.initialize()
    logger.info("✓ LLM generator initialized")

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(retriever, generator)
    logger.info("✓ RAG pipeline ready\n")

    # Initialize evaluation harness
    logger.info(f"Loading evaluation questions from {questions_file}...")
    harness = EvaluationHarness(questions_file)
    harness.load_questions()

    # Limit questions if specified
    if num_questions is not None:
        harness.questions = harness.questions[:num_questions]
        logger.info(f"Limited to {num_questions} questions for evaluation")

    # Run evaluation
    logger.info("\n" + "="*60)
    logger.info("Starting Evaluation")
    logger.info("="*60 + "\n")

    results = harness.run_evaluation(
        config_name=exp_name,
        rag_pipeline=rag_pipeline
    )

    # Print summary
    harness.print_summary(results)

    # Save results
    if output_file is None:
        output_file = f"results/{exp_name}_evaluation.csv"

    harness.save_results(results, output_file)

    logger.info(f"\n✓ Evaluation complete!")
    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML file (defaults to baseline.yaml)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/questions/eval_questions.json",
        help="Path to questions JSON file"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Path to chunks JSON file (auto-derived from config if not provided)"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to FAISS index file (auto-derived from config if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)"
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Use experiment-specific index ({experiment_name}_index.faiss). If not set, uses baseline_index.faiss. Chunks always use baseline."
    )
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        questions_file=args.questions,
        chunks_file=args.chunks,
        index_file=args.index,
        output_file=args.output,
        num_questions=args.num_questions,
        artifacts_name=args.artifacts
    )


if __name__ == "__main__":
    main()
