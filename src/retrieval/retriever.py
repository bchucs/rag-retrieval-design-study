"""Main retrieval interface

Orchestrates the complete retrieval pipeline.
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass, field
import logging

from .embeddings import EmbeddingModel
from .indexing import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval

    IMPORTANT: Always load from config files using utils.config.get_retrieval_config()
    Do not use default values - all values should come from YAML configs
    """
    top_k: int  # number of documents to retrieve
    distance_threshold: Optional[float] = None  # optional distance cutoff


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    query: str
    documents: List[Dict[str, Any]]
    distances: List[float]
    chunk_ids: List[str]
    latency_ms: float
    num_results: int = field(init=False)

    def __post_init__(self):
        self.num_results = len(self.documents)


class Retriever:
    """Main retrieval interface"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: VectorIndex,
        config: RetrievalConfig,
        chunks: List[Dict[str, Any]]
    ):
        """Initialize retriever

        Args:
            embedding_model: Model for generating embeddings
            vector_index: Vector index for similarity search
            config: Retrieval configuration
            chunks: List of all document chunks with metadata
        """
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.config = config
        self.chunks = chunks

        # Create lookup dict for fast chunk access by index
        self.chunk_lookup = {i: chunk for i, chunk in enumerate(chunks)}

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant documents for a query

        Args:
            query: Query string

        Returns:
            RetrievalResult with documents, distances, and latency
        """
        start_time = time.time()

        # 1. Embed query
        query_embedding = self.embedding_model.embed_query(query, normalize=True)

        # 2. Search index
        distances, indices = self.vector_index.search(
            query_embedding,
            k=self.config.top_k,
            distance_threshold=self.config.distance_threshold
        )

        # 3. Fetch documents from indices
        documents = []
        chunk_ids = []

        for idx in indices:
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            if idx in self.chunk_lookup:
                chunk = self.chunk_lookup[idx]
                documents.append(chunk)
                chunk_ids.append(chunk.get('chunk_id', f'chunk_{idx}'))
            else:
                logger.warning(f"Index {idx} not found in chunk lookup")

        # 4. Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            documents=documents,
            distances=distances.tolist(),
            chunk_ids=chunk_ids,
            latency_ms=latency_ms
        )

    def batch_retrieve(
        self,
        queries: List[str],
        show_progress: bool = False
    ) -> List[RetrievalResult]:
        """Retrieve for multiple queries

        Args:
            queries: List of query strings
            show_progress: Whether to show progress

        Returns:
            List of RetrievalResults
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            queries = tqdm(queries, desc="Retrieving")

        for query in queries:
            result = self.retrieve(query)
            results.append(result)

        return results

    def get_unique_doc_ids(self, result: RetrievalResult) -> List[str]:
        """Get unique document IDs from retrieval result

        Args:
            result: RetrievalResult

        Returns:
            List of unique doc_ids
        """
        doc_ids = []
        seen = set()

        for doc in result.documents:
            doc_id = doc.get('doc_id', doc.get('original_doc', {}).get('doc_id'))
            if doc_id and doc_id not in seen:
                doc_ids.append(doc_id)
                seen.add(doc_id)

        return doc_ids