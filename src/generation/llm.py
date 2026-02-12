"""LLM interface for answer generation

Uses a fixed LLM throughout all experiments.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for LLM generation

    IMPORTANT: Always load from config files using utils.config.get_generation_config()
    Do not use default values - all values should come from YAML configs
    """
    model_name: str  # e.g., "gpt-4o-mini"
    temperature: float = 0.0  # sensible default for reproducibility
    max_tokens: int = 500  # sensible default
    api_key: Optional[str] = None  # can also be set via OPENAI_API_KEY env var


@dataclass
class GenerationResult:
    """Result from LLM generation"""
    query: str
    answer: str
    prompt: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMGenerator:
    """Interface for LLM-based answer generation"""

    def __init__(self, generation_config: GenerationConfig):
        """Initialize LLM generator

        Args:
            generation_config: Generation configuration
        """
        self.generation_config = generation_config
        self.client = None

    def initialize(self):
        """Initialize the LLM client"""
        api_key = self.generation_config.api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key in GenerationConfig"
            )

        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {self.generation_config.model_name}")

    def generate_answer(
        self,
        query: str,
        context_documents: List[Dict[str, Any]]
    ) -> GenerationResult:
        """Generate an answer given query and retrieved context

        Args:
            query: User query
            context_documents: Retrieved documents to use as context

        Returns:
            GenerationResult with answer and metadata
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")

        start_time = time.time()

        # Build prompt with context
        prompt = self._build_prompt(query, context_documents)

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.generation_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable assistant specializing in computer science and machine learning research. "
                                   "Answer questions based on the provided context from arXiv research papers. "
                                   "Focus on technical accuracy and cite specific concepts, methods, or findings from the papers. "
                                   "If the context doesn't contain enough information to answer the question fully, "
                                   "clearly state what information is missing. Do not make up or infer information beyond what is provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.generation_config.temperature,
                max_tokens=self.generation_config.max_tokens,
            )

            answer = response.choices[0].message.content
            usage = response.usage

            latency_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                query=query,
                answer=answer,
                prompt=prompt,
                model=self.generation_config.model_name,
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _build_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Build prompt template with query and context

        Args:
            query: User query
            documents: Context documents

        Returns:
            Formatted prompt string
        """
        if not documents:
            return f"Question: {query}\n\nNo research papers provided in context."

        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents):
            # Extract text from chunk
            text = doc.get('text', '')

            # Get metadata from original document if available
            original_doc = doc.get('original_doc', {})
            title = original_doc.get('title', 'Unknown')
            categories = original_doc.get('categories', 'N/A')

            context_parts.append(
                f"[Paper {i+1}]\n"
                f"Title: {title}\n"
                f"Categories: {categories}\n"
                f"Content: {text}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are answering a question about computer science and machine learning research based on arXiv papers.

Research Papers:
{context}

Question: {query}

Provide a clear, technical answer based on the information in the papers above. Reference specific papers when relevant. If the papers don't contain sufficient information to fully answer the question, state what's missing.

Answer:"""
        return prompt

    def batch_generate(
        self,
        queries: List[str],
        context_documents_list: List[List[Dict[str, Any]]],
        show_progress: bool = True
    ) -> List[GenerationResult]:
        """Generate answers for multiple queries

        Args:
            queries: List of queries
            context_documents_list: List of document lists (one per query)
            show_progress: Whether to show progress bar

        Returns:
            List of GenerationResults
        """
        if len(queries) != len(context_documents_list):
            raise ValueError("Number of queries must match number of context document lists")

        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(zip(queries, context_documents_list), total=len(queries), desc="Generating answers")
        else:
            iterator = zip(queries, context_documents_list)

        for query, context_docs in iterator:
            result = self.generate_answer(query, context_docs)
            results.append(result)

        return results