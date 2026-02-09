"""Evaluation harness for running experiments

Coordinates running queries across different configurations and collecting results.
"""

from typing import List, Dict, Any
import csv
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from openai import OpenAI

from .metrics import retrieval_recall, answer_correctness_score, hallucination_detection

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuestion:
    """A single evaluation question with metadata"""
    question_id: str
    question: str
    gold_doc_ids: List[str]
    reference_answer: str = None


@dataclass
class EvaluationResult:
    """Result for a single question evaluation"""
    question_id: str
    question: str
    config_name: str
    retrieved_doc_ids: List[str]
    retrieval_recall: float
    answer: str
    reference_answer: str
    answer_correctness: float
    has_hallucination: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class EvaluationHarness:
    """Runs evaluation across multiple configurations"""

    def __init__(self, questions_path: str):
        """Initialize evaluation harness

        Args:
            questions_path: Path to questions file
        """
        self.questions_path = questions_path
        self.questions: List[EvaluationQuestion] = []

    def load_questions(self):
        """Load evaluation questions from file"""
        with open(self.questions_path, 'r') as f:
            questions_data = json.load(f)

        self.questions = []
        for q_data in questions_data:
            question = EvaluationQuestion(
                question_id=q_data['question_id'],
                question=q_data['question'],
                gold_doc_ids=q_data['gold_doc_ids'],
                reference_answer=q_data.get('answer', '')
            )
            self.questions.append(question)

        logger.info(f"Loaded {len(self.questions)} questions from {self.questions_path}")

    def run_evaluation(
        self,
        config_name: str,
        rag_pipeline,
        client: OpenAI = None
    ) -> List[EvaluationResult]:
        """Run evaluation for a specific configuration

        Args:
            config_name: Name of the configuration being tested
            rag_pipeline: RAGPipeline instance
            client: OpenAI client for evaluation metrics

        Returns:
            List of evaluation results
        """
        if not self.questions:
            raise RuntimeError("No questions loaded. Call load_questions() first.")

        if client is None:
            client = OpenAI()

        logger.info(f"Running evaluation for config: {config_name}")
        logger.info(f"Evaluating {len(self.questions)} questions")

        results = []

        for i, question in enumerate(self.questions):
            logger.info(f"Processing question {i+1}/{len(self.questions)}: {question.question_id}")

            # Run RAG pipeline
            rag_result = rag_pipeline.query(question.question, log_details=False)

            # Get unique doc IDs from retrieved documents
            retrieved_doc_ids = rag_pipeline.retriever.get_unique_doc_ids(
                rag_result.retrieval_result
            )

            # Calculate retrieval recall
            recall = retrieval_recall(retrieved_doc_ids, question.gold_doc_ids)

            # Evaluate answer correctness
            correctness = answer_correctness_score(
                generated_answer=rag_result.answer,
                reference_answer=question.reference_answer,
                question=question.question,
                client=client
            )

            # Detect hallucinations
            has_hallucination = hallucination_detection(
                answer=rag_result.answer,
                context_documents=rag_result.retrieval_result.documents,
                question=question.question,
                client=client
            )

            # Create result
            result = EvaluationResult(
                question_id=question.question_id,
                question=question.question,
                config_name=config_name,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieval_recall=recall,
                answer=rag_result.answer,
                reference_answer=question.reference_answer,
                answer_correctness=correctness,
                has_hallucination=has_hallucination,
                retrieval_latency_ms=rag_result.retrieval_result.latency_ms,
                generation_latency_ms=rag_result.generation_result.latency_ms,
                total_latency_ms=rag_result.total_latency_ms
            )

            results.append(result)

            logger.info(f"  Recall: {recall:.2f}, Correctness: {correctness:.1f}/2, "
                       f"Hallucination: {has_hallucination}")

        logger.info(f"\nEvaluation complete for {config_name}")
        logger.info(f"Processed {len(results)} questions")

        return results

    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results to CSV

        Args:
            results: List of evaluation results
            output_path: Path to output CSV file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if not results:
                return

            # Convert to dict and handle list fields
            fieldnames = list(asdict(results[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = asdict(result)
                # Convert list to string for CSV
                row['retrieved_doc_ids'] = '|'.join(row['retrieved_doc_ids'])
                writer.writerow(row)

        logger.info(f"Saved {len(results)} results to {output_path}")

    def print_summary(self, results: List[EvaluationResult]):
        """Print summary statistics for evaluation results

        Args:
            results: List of evaluation results
        """
        if not results:
            logger.info("No results to summarize")
            return

        # Calculate aggregate statistics
        avg_recall = sum(r.retrieval_recall for r in results) / len(results)
        avg_correctness = sum(r.answer_correctness for r in results) / len(results)
        hallucination_rate = sum(r.has_hallucination for r in results) / len(results)
        avg_retrieval_latency = sum(r.retrieval_latency_ms for r in results) / len(results)
        avg_generation_latency = sum(r.generation_latency_ms for r in results) / len(results)
        avg_total_latency = sum(r.total_latency_ms for r in results) / len(results)

        # Perfect recall count
        perfect_recall_count = sum(1 for r in results if r.retrieval_recall == 1.0)
        fully_correct_count = sum(1 for r in results if r.answer_correctness == 2.0)

        logger.info("\n" + "="*60)
        logger.info(f"EVALUATION SUMMARY - {results[0].config_name}")
        logger.info("="*60)
        logger.info(f"Total questions: {len(results)}")
        logger.info(f"\nRetrieval Performance:")
        logger.info(f"  Average recall: {avg_recall:.3f}")
        logger.info(f"  Perfect recall (1.0): {perfect_recall_count}/{len(results)} ({perfect_recall_count/len(results)*100:.1f}%)")
        logger.info(f"\nAnswer Quality:")
        logger.info(f"  Average correctness: {avg_correctness:.2f}/2.0")
        logger.info(f"  Fully correct (2.0): {fully_correct_count}/{len(results)} ({fully_correct_count/len(results)*100:.1f}%)")
        logger.info(f"  Hallucination rate: {hallucination_rate:.1%}")
        logger.info(f"\nLatency:")
        logger.info(f"  Average retrieval: {avg_retrieval_latency:.2f}ms")
        logger.info(f"  Average generation: {avg_generation_latency:.2f}ms")
        logger.info(f"  Average total: {avg_total_latency:.2f}ms")
        logger.info("="*60 + "\n")
