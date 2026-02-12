"""End-to-end RAG pipeline

Combines retrieval and generation into a complete question-answering system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging

from retrieval.retriever import Retriever, RetrievalResult
from generation.llm import LLMGenerator, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Complete RAG pipeline result"""
    query: str
    answer: str
    retrieval_result: RetrievalResult
    generation_result: GenerationResult
    total_latency_ms: float
    num_documents_used: int


class RAGPipeline:
    """End-to-end RAG pipeline combining retrieval and generation"""

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator
    ):
        """Initialize RAG pipeline

        Args:
            retriever: Retriever instance for document retrieval
            generator: LLM generator for answer generation
        """
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        question: str,
        log_details: bool = True
    ) -> RAGResult:
        """Process a question through the complete RAG pipeline

        Args:
            question: User question
            log_details: Whether to log detailed information

        Returns:
            RAGResult with answer and metadata
        """
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        if log_details:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing query: {question}")
            logger.info(f"{'='*60}")
            logger.info("\n[1/2] Retrieving relevant documents...")

        retrieval_result = self.retriever.retrieve(question)

        if log_details:
            logger.info(f"Retrieved {retrieval_result.num_results} documents in {retrieval_result.latency_ms:.2f}ms")
            logger.info("\nTop retrieved documents:")
            for i, (doc, score) in enumerate(zip(
                retrieval_result.documents[:3],
                retrieval_result.distances[:3]
            )):
                title = doc.get('original_doc', {}).get('title', 'N/A')
                logger.info(f"  {i+1}. [score={score:.3f}] {title[:80]}...")

        # Step 2: Generate answer using retrieved context
        if log_details:
            logger.info("\n[2/2] Generating answer with LLM...")

        generation_result = self.generator.generate_answer(
            question,
            retrieval_result.documents
        )

        if log_details:
            logger.info(f"Generated answer in {generation_result.latency_ms:.2f}ms")
            logger.info(f"Tokens: {generation_result.total_tokens} "
                       f"(prompt: {generation_result.prompt_tokens}, "
                       f"completion: {generation_result.completion_tokens})")

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000

        result = RAGResult(
            query=question,
            answer=generation_result.answer,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            total_latency_ms=total_latency_ms,
            num_documents_used=len(retrieval_result.documents)
        )

        if log_details:
            logger.info(f"\n{'─'*60}")
            logger.info("ANSWER:")
            logger.info(f"{'─'*60}")
            logger.info(result.answer)
            logger.info(f"\n{'─'*60}")
            logger.info(f"Total latency: {total_latency_ms:.2f}ms")
            logger.info(f"{'─'*60}\n")

        return result

    def batch_query(
        self,
        questions: List[str],
        show_progress: bool = False
    ) -> List[RAGResult]:
        """Process multiple questions through the pipeline

        Args:
            questions: List of questions
            show_progress: Whether to show progress bar

        Returns:
            List of RAGResults
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            questions = tqdm(questions, desc="Processing questions")

        for question in questions:
            result = self.query(question, log_details=False)
            results.append(result)

        logger.info(f"\nProcessed {len(results)} questions")
        logger.info(f"Average latency: {sum(r.total_latency_ms for r in results) / len(results):.2f}ms")

        return results

    def get_summary_stats(self, results: List[RAGResult]) -> Dict[str, Any]:
        """Get summary statistics from a batch of results

        Args:
            results: List of RAGResults

        Returns:
            Dictionary of summary statistics
        """
        if not results:
            return {}

        return {
            'num_queries': len(results),
            'avg_total_latency_ms': sum(r.total_latency_ms for r in results) / len(results),
            'avg_retrieval_latency_ms': sum(r.retrieval_result.latency_ms for r in results) / len(results),
            'avg_generation_latency_ms': sum(r.generation_result.latency_ms for r in results) / len(results),
            'avg_documents_used': sum(r.num_documents_used for r in results) / len(results),
            'total_tokens': sum(r.generation_result.total_tokens for r in results),
            'total_prompt_tokens': sum(r.generation_result.prompt_tokens for r in results),
            'total_completion_tokens': sum(r.generation_result.completion_tokens for r in results),
        }
