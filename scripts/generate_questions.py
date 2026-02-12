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

    Format your response as a JSON array of objects with these fields:
    - "question": The question text
    - "answer": A concise answer (2-3 sentences) based on the abstract
    - "doc_id": "{doc_id}"

    Return ONLY the JSON array, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating evaluation questions for machine learning research papers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # Try to parse JSON
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        questions = json.loads(content)

        # Add metadata
        for q in questions:
            q['gold_doc_ids'] = [doc_id]
            q['title'] = title

        return questions

    except Exception as e:
        logger.error(f"Error generating questions for doc {doc_id}: {e}")
        return []


def generate_question_set(
    chunks_file: str,
    output_file: str,
    target_questions: int = 50,
    questions_per_doc: int = 2
) -> List[Dict[str, Any]]:
    """Generate a set of evaluation questions

    Args:
        chunks_file: Path to chunks file
        output_file: Path to save questions
        target_questions: Target number of questions
        questions_per_doc: Questions to generate per document

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

    # Calculate how many documents we need
    num_docs_needed = (target_questions + questions_per_doc - 1) // questions_per_doc
    num_docs_needed = min(num_docs_needed, len(documents))

    logger.info(f"Generating {target_questions} questions from {num_docs_needed} documents")
    logger.info(f"Generating {questions_per_doc} questions per document")

    all_questions = []

    for i, doc in enumerate(documents[:num_docs_needed]):
        logger.info(f"Processing document {i+1}/{num_docs_needed}: {doc.get('title', 'Unknown')[:60]}...")

        questions = generate_questions_for_document(client, doc, questions_per_doc)
        all_questions.extend(questions)

        logger.info(f"  Generated {len(questions)} questions (total: {len(all_questions)})")

        if len(all_questions) >= target_questions:
            break

    # Trim to target
    all_questions = all_questions[:target_questions]

    # Add question IDs
    for i, q in enumerate(all_questions):
        q['question_id'] = f"q_{i:03d}"

    # Save questions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_questions, f, indent=2)

    logger.info(f"\nGenerated {len(all_questions)} questions")
    logger.info(f"Saved to: {output_path}")

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
    args = parser.parse_args()

    questions = generate_question_set(
        chunks_file=args.chunks_file,
        output_file=args.output,
        target_questions=args.num_questions,
        questions_per_doc=args.questions_per_doc
    )

    # Print sample questions
    logger.info("\nSample questions:")
    for q in questions[:3]:
        logger.info(f"\nQ: {q['question']}")
        logger.info(f"A: {q['answer']}")
        logger.info(f"Source: {q.get('title', 'N/A')[:60]}...")


if __name__ == "__main__":
    main()
