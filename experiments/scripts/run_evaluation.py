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
from retrieval.indexing import VectorIndex, IndexConfig
from retrieval.retriever import Retriever, RetrievalConfig
from generation.llm import LLMGenerator, GenerationConfig
from rag_pipeline import RAGPipeline
from evaluation.harness import EvaluationHarness
from utils.config import load_config, get_embedding_model_name, get_retrieval_config, get_generation_config

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
    config = load_config()
    embedding_model_name = get_embedding_model_name(config)
    retrieval_cfg = get_retrieval_config(config)
    generation_cfg = get_generation_config(config)
    logger.info(f"Config: {config_name}")
    logger.info(f"  Embedding: {embedding_model_name}")
    logger.info(f"  Generation: {generation_cfg['model_name']}")
    logger.info(f"  Top-k: {retrieval_cfg['top_k']}")

    # Load artifacts
    logger.info("\nLoading baseline artifacts...")

    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    logger.info(f"✓ Loaded {len(chunks)} chunks")

    # Load embedding model
    logger.info(f"Loading embedding model...")
    embedding_model = EmbeddingModel(embedding_model_name)
    embedding_model.load()
    logger.info(f"✓ Model loaded")

    # Load index
    logger.info("Loading vector index...")
    index_config = IndexConfig(index_type="flat")
    vector_index = VectorIndex(index_config, embedding_model.embedding_dim)
    vector_index.load(str(index_file))
    logger.info(f"✓ Index loaded ({vector_index.n_vectors} vectors)")

    # Create retriever
    retrieval_config = RetrievalConfig(
        top_k=retrieval_cfg['top_k'],
        distance_threshold=retrieval_cfg.get('distance_threshold')
    )
    retriever = Retriever(embedding_model, vector_index, retrieval_config, chunks)
    logger.info("✓ Retriever initialized")

    # Initialize LLM generator
    logger.info("Initializing LLM generator...")
    generation_config = GenerationConfig(
        model_name=generation_cfg['model_name'],
        temperature=generation_cfg['temperature'],
        max_tokens=generation_cfg['max_tokens']
    )
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
