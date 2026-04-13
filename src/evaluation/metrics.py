"""Evaluation metrics for RAG systems

Implements metrics for retrieval quality and answer correctness.
Uses ragas library for factual correctness, faithfulness, and context precision.
Custom implementations for retrieval recall and abstention scoring.
"""

from typing import List, Dict, Any, Optional
import logging
from openai import OpenAI

from ragas.metrics.collections import FactualCorrectness, Faithfulness, ContextPrecision
from ragas.llms import llm_factory

logger = logging.getLogger(__name__)

# Lazy singleton for ragas LLM wrapper
_ragas_llm = None


def _get_ragas_llm():
    """Lazily initialize the ragas LLM wrapper (gpt-4o-mini via OpenAI)."""
    global _ragas_llm
    if _ragas_llm is None:
        from openai import AsyncOpenAI
        _ragas_llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
    return _ragas_llm


def retrieval_recall(
    retrieved_doc_ids: List[str],
    gold_doc_ids: List[str]
) -> float:
    """Calculate retrieval recall

    Args:
        retrieved_doc_ids: IDs of retrieved documents
        gold_doc_ids: IDs of gold/relevant documents

    Returns:
        Recall score (fraction of gold docs retrieved)
    """
    if not gold_doc_ids:
        return 0.0

    retrieved_set = set(retrieved_doc_ids)
    gold_set = set(gold_doc_ids)

    hits = len(retrieved_set.intersection(gold_set))
    return hits / len(gold_set)


def factual_correctness_score(
    generated_answer: str,
    reference_answer: str,
    question: str,
) -> float:
    """Evaluate answer correctness using ragas FactualCorrectness.

    Decomposes both response and reference into atomic claims, then computes
    an F1-style score based on claim overlap.

    Args:
        generated_answer: Generated answer from RAG system
        reference_answer: Reference/gold answer
        question: The original question

    Returns:
        Correctness score in [0, 1]. Higher is better.
    """
    llm = _get_ragas_llm()
    scorer = FactualCorrectness(llm=llm)
    try:
        result = scorer.score(response=generated_answer, reference=reference_answer)
        return float(result)
    except Exception as e:
        logger.error(f"Error in factual correctness evaluation: {e}")
        return 0.0


def faithfulness_score(
    answer: str,
    context_documents: List[Dict[str, Any]],
    question: str,
) -> float:
    """Evaluate answer faithfulness using ragas Faithfulness.

    Checks what fraction of claims in the answer are supported by the
    retrieved context documents.

    Args:
        answer: Generated answer
        context_documents: Retrieved context documents
        question: The original question

    Returns:
        Faithfulness score in [0, 1]. 1.0 = fully grounded, 0.0 = fully hallucinated.
    """
    llm = _get_ragas_llm()
    scorer = Faithfulness(llm=llm)
    contexts = [doc.get('text', '') for doc in context_documents]
    try:
        result = scorer.score(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(result)
    except Exception as e:
        logger.error(f"Error in faithfulness evaluation: {e}")
        return 0.0


def context_precision_score(
    question: str,
    reference_answer: str,
    context_documents: List[Dict[str, Any]],
) -> float:
    """Evaluate retrieval ranking quality using ragas ContextPrecision.

    Measures whether relevant documents are ranked higher than irrelevant
    ones in the retrieved set.

    Args:
        question: The original question
        reference_answer: Reference/gold answer
        context_documents: Retrieved context documents

    Returns:
        Context precision score in [0, 1]. Higher means relevant docs ranked higher.
    """
    llm = _get_ragas_llm()
    scorer = ContextPrecision(llm=llm)
    contexts = [doc.get('text', '') for doc in context_documents]
    try:
        result = scorer.score(
            user_input=question, reference=reference_answer, retrieved_contexts=contexts
        )
        return float(result)
    except Exception as e:
        logger.error(f"Error in context precision evaluation: {e}")
        return 0.0


def abstention_score(
    answer: str,
    client: Optional[OpenAI] = None
) -> float:
    """Evaluate if the system properly abstained from answering an unanswerable question

    Args:
        answer: Generated answer from RAG system
        client: OpenAI client (optional, will create if not provided)

    Returns:
        Abstention score on 0-2 scale:
        - 0: Confidently answered (should have abstained)
        - 1: Partially hedged or expressed uncertainty
        - 2: Clearly refused to answer / stated insufficient information
    """
    if client is None:
        client = OpenAI()

    prompt = f"""You are evaluating whether a question-answering system properly abstained from answering a question it should not answer.

Generated Answer: {answer}

Rate how well the system abstained on a scale of 0-2:
- 0: Provided a confident answer with specific factual claims (should have refused)
- 1: Attempted to answer but heavily hedged or expressed significant uncertainty
- 2: Clearly refused to answer or stated it doesn't have sufficient information

Examples of score 2:
- "I don't have enough information to answer this question."
- "I cannot answer this based on the provided documents."
- "This information is not available in my context."

Examples of score 1:
- "I'm not entirely sure, but it might be X..."
- "Based on limited information, possibly Y, though I cannot confirm."

Examples of score 0:
- "The answer is X because Y."
- "According to the research, Z is the case."

Respond with ONLY a single number: 0, 1, or 2."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator for question-answering systems. Provide objective ratings."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=10,
        )

        content = response.choices[0].message.content.strip()

        # Parse score
        try:
            score = float(content)
            if score not in [0, 1, 2]:
                logger.warning(f"LLM returned invalid abstention score: {content}, defaulting to 0")
                score = 0.0
        except ValueError:
            logger.warning(f"Could not parse abstention score: {content}, defaulting to 0")
            score = 0.0

        return score

    except Exception as e:
        logger.error(f"Error in abstention score evaluation: {e}")
        return 0.0