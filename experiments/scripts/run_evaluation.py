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
    config_name: str = "baseline",
    questions_file: str = "data/questions/eval_questions.json",
    chunks_file: str = "data/processed/baseline_chunks.json",
    index_file: str = "data/processed/baseline_index.faiss",
    output_file: str = None,
    num_questions: int = None
):
    """Run evaluation on baseline RAG system

    Args:
        config_name: Name of configuration being tested
        questions_file: Path to questions JSON file
        chunks_file: Path to chunks JSON file
        index_file: Path to FAISS index file
        output_file: Path to save results CSV (optional)
        num_questions: Number of questions to evaluate (None for all)
    """
    logger.info("="*60)
    logger.info("RAG System Evaluation")
    logger.info("="*60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Load configuration
    logger.info("\nLoading configuration...")
    config = config_utils.load_config()
    emb_config = config_utils.get_embedding_config(config)
    retrieval_config = config_utils.get_retrieval_config(config)
    generation_config = config_utils.get_generation_config(config)
    index_config = config_utils.get_index_config(config)

    logger.info(f"Config: {config_name}")
    logger.info(f"  Embedding: {emb_config['model_name']}")
    logger.info(f"  Generation: {generation_config.model_name}")
    logger.info(f"  Top-k: {retrieval_config.top_k}")

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
        config_name=config_name,
        rag_pipeline=rag_pipeline
    )

    # Print summary
    harness.print_summary(results)

    # Save results
    if output_file is None:
        output_file = f"results/{config_name}_evaluation.csv"

    harness.save_results(results, output_file)

    logger.info(f"\n✓ Evaluation complete!")
    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--config-name",
        type=str,
        default="baseline",
        help="Name of configuration being tested"
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
        default="data/processed/baseline_chunks.json",
        help="Path to chunks JSON file"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/processed/baseline_index.faiss",
        help="Path to FAISS index file"
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
    args = parser.parse_args()

    run_evaluation(
        config_name=args.config_name,
        questions_file=args.questions,
        chunks_file=args.chunks,
        index_file=args.index,
        output_file=args.output,
        num_questions=args.num_questions
    )


if __name__ == "__main__":
    main()
