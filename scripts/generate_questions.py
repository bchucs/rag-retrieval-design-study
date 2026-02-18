#!/usr/bin/env python3
"""Generate evaluation questions from arXiv papers

This script uses an LLM to generate factual questions from the corpus for evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import os
import logging
import argparse
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(chunks_file: str) -> List[Dict[str, Any]]:
    """Load document chunks

    Args:
        chunks_file: Path to chunks JSON file

    Returns:
        List of document chunks
    """
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def get_unique_documents(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get unique documents from chunks

    Args:
        chunks: List of chunks

    Returns:
        List of unique documents
    """
    docs_dict = {}
    for chunk in chunks:
        doc_id = chunk.get('doc_id')
        if doc_id and doc_id not in docs_dict:
            docs_dict[doc_id] = chunk.get('original_doc', {})

    documents = list(docs_dict.values())
    logger.info(f"Found {len(documents)} unique documents")
    return documents


def generate_questions_for_document(
    client: OpenAI,
    document: Dict[str, Any],
    num_questions: int = 2
) -> List[Dict[str, Any]]:
    """Generate factual questions for a single document

    Args:
        client: OpenAI client
        document: Document to generate questions for
        num_questions: Number of questions to generate

    Returns:
        List of question dictionaries
    """
    title = document.get('title', 'Unknown')
    abstract = document.get('abstract', '')
    doc_id = document.get('doc_id', 'unknown')

    prompt = f"""Given the following research paper abstract, generate {num_questions} factual questions that can be answered using ONLY the information in this abstract.

    Title: {title}

    Abstract: {abstract}

    Requirements:
    1. Questions should be specific and factual
    2. Questions should test understanding of key concepts, methods, or findings
    3. Answers should be directly found in the abstract
    4. Questions should be relevant for evaluating a retrieval-augmented generation system
    5. Avoid yes/no questions

    Format your response as a JSON object with a "questions" array containing objects with these fields:
    - "question": The question text
    - "answer": A concise answer (2-3 sentences) based on the abstract
    - "doc_id": "{doc_id}"

    Return ONLY the JSON object in this format: {{"questions": [...]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating evaluation questions for machine learning research papers. You must return valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()

        # Try to parse JSON
        # Remove markdown code blocks if present (shouldn't be needed with response_format)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON parsing error for doc {doc_id}: {je}")
            logger.error(f"Raw content (first 500 chars): {content[:500]}")
            logger.error(f"Raw content (last 500 chars): {content[-500:]}")
            return []

        # Extract questions array from the response object
        questions = result.get('questions', result if isinstance(result, list) else [])

        # Add metadata
        for q in questions:
            q['gold_doc_ids'] = [doc_id]
            q['title'] = title
            q['answerable'] = True

        return questions

    except Exception as e:
        logger.error(f"Error generating questions for doc {doc_id}: {e}")
        return []


def generate_unanswerable_questions(
    client: OpenAI,
    documents: List[Dict[str, Any]],
    num_questions: int = 10
) -> List[Dict[str, Any]]:
    """Generate questions that cannot be answered by the corpus

    Args:
        client: OpenAI client
        documents: Sample documents to understand domain
        num_questions: Number of unanswerable questions to generate

    Returns:
        List of unanswerable question dictionaries
    """
    # Get sample titles to understand domain
    sample_titles = [doc.get('title', '') for doc in documents[:5]]
    domain_context = "\n".join(f"- {title}" for title in sample_titles)

    prompt = f"""You are generating evaluation questions for a computer science research paper retrieval system.

The corpus contains papers like:
{domain_context}

Generate {num_questions} questions that:
1. Are relevant to computer science/machine learning research (seem like questions someone might ask)
2. CANNOT be answered by typical research papers in this domain
3. Are specific enough that it's clear they require information not in academic papers
4. Would test if the system admits uncertainty vs fabricating answers

Examples of good unanswerable questions:
- "What is the current price of NVIDIA A100 GPUs?"
- "Which professor at Stanford won the Turing Award in 2023?"
- "What time does the NeurIPS 2024 conference registration desk open?"
- "How many GitHub stars does the Hugging Face transformers repository have today?"

Format your response as a JSON object with a "questions" array containing objects with these fields:
- "question": The question text
- "answer": "I don't have enough information in the provided documents to answer this question." (exactly this text)

Return ONLY the JSON object in this format: {{"questions": [...]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating evaluation questions for retrieval systems. You must return valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.9,
            max_tokens=2500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON parsing error: {je}")
            logger.error(f"Raw content (first 500 chars): {content[:500]}")
            logger.error(f"Raw content (last 500 chars): {content[-500:]}")
            return []

        # Extract questions array from the response object
        questions = result.get('questions', result if isinstance(result, list) else [])

        # Add metadata
        for q in questions:
            q['gold_doc_ids'] = []  # No correct documents exist
            q['title'] = None
            q['answerable'] = False

        return questions

    except Exception as e:
        logger.error(f"Error generating unanswerable questions: {e}")
        return []


def generate_question_set(
    chunks_file: str,
    output_file: str,
    target_questions: int = 50,
    questions_per_doc: int = 2,
    unanswerable_ratio: float = 0.2
) -> List[Dict[str, Any]]:
    """Generate a set of evaluation questions

    Args:
        chunks_file: Path to chunks file
        output_file: Path to save questions
        target_questions: Target number of total questions
        questions_per_doc: Questions to generate per document
        unanswerable_ratio: Fraction of questions that should be unanswerable (0.0-1.0)

    Returns:
        List of generated questions
    """
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=api_key)

    # Load documents
    chunks = load_documents(chunks_file)
    documents = get_unique_documents(chunks)

    # Calculate split between answerable and unanswerable
    num_unanswerable = int(target_questions * unanswerable_ratio)
    num_answerable = target_questions - num_unanswerable

    logger.info(f"Generating {target_questions} total questions:")
    logger.info(f"  Answerable: {num_answerable}")
    logger.info(f"  Unanswerable: {num_unanswerable}")

    # Generate answerable questions
    logger.info(f"\n--- Generating answerable questions ---")
    num_docs_needed = (num_answerable + questions_per_doc - 1) // questions_per_doc
    num_docs_needed = min(num_docs_needed, len(documents))
    logger.info(f"Using {num_docs_needed} documents, {questions_per_doc} questions per document")

    answerable_questions = []

    for i, doc in enumerate(documents[:num_docs_needed]):
        logger.info(f"Processing document {i+1}/{num_docs_needed}: {doc.get('title', 'Unknown')[:60]}...")

        questions = generate_questions_for_document(client, doc, questions_per_doc)
        answerable_questions.extend(questions)

        logger.info(f"  Generated {len(questions)} questions (total: {len(answerable_questions)})")

        if len(answerable_questions) >= num_answerable:
            break

    # Trim to target
    answerable_questions = answerable_questions[:num_answerable]
    logger.info(f"Generated {len(answerable_questions)} answerable questions")

    # Generate unanswerable questions
    logger.info(f"\n--- Generating unanswerable questions ---")
    unanswerable_questions = []
    if num_unanswerable > 0:
        unanswerable_questions = generate_unanswerable_questions(
            client, documents, num_unanswerable
        )
        logger.info(f"Generated {len(unanswerable_questions)} unanswerable questions")

    # Combine and shuffle
    all_questions = answerable_questions + unanswerable_questions
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(all_questions)

    # Add question IDs
    for i, q in enumerate(all_questions):
        q['question_id'] = f"q_{i:03d}"

    # Save questions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_questions, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Generated {len(all_questions)} total questions")
    logger.info(f"  Answerable: {len(answerable_questions)}")
    logger.info(f"  Unanswerable: {len(unanswerable_questions)}")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"{'='*60}")

    return all_questions


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation questions")
    parser.add_argument(
        "--chunks-file",
        type=str,
        default="data/processed/baseline_chunks.json",
        help="Path to chunks JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/questions/eval_questions.json",
        help="Path to save questions"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Target number of questions to generate"
    )
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=2,
        help="Questions to generate per document"
    )
    parser.add_argument(
        "--unanswerable-ratio",
        type=float,
        default=0.2,
        help="Fraction of questions that should be unanswerable (0.0-1.0)"
    )
    args = parser.parse_args()

    questions = generate_question_set(
        chunks_file=args.chunks_file,
        output_file=args.output,
        target_questions=args.num_questions,
        questions_per_doc=args.questions_per_doc,
        unanswerable_ratio=args.unanswerable_ratio
    )

    # Print sample questions
    logger.info("\nSample questions:")
    for q in questions[:5]:
        q_type = "ANSWERABLE" if q['answerable'] else "UNANSWERABLE"
        logger.info(f"\n[{q_type}] Q: {q['question']}")
        logger.info(f"A: {q['answer']}")
        if q['answerable']:
            logger.info(f"Source: {q.get('title', 'N/A')[:60]}...")
        else:
            logger.info(f"Source: (none - out of corpus)")


if __name__ == "__main__":
    main()
