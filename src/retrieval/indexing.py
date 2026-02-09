"""Vector indexing strategies

Supports flat and clustered (IVF) indexing approaches.
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import faiss
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for vector index

    Note: Default values are provided, but load from experiments/configs/baseline.yaml in practice
    """
    index_type: str = "flat"  # "flat" or "ivf" - override with config
    n_clusters: int = 100  # for IVF index - override with config
    nprobe: int = 10  # number of clusters to search in IVF


class VectorIndex:
    """Vector index for similarity search"""

    def __init__(self, config: IndexConfig, embedding_dim: int):
        """Initialize vector index

        Args:
            config: Index configuration
            embedding_dim: Dimension of embedding vectors
        """
        self.config = config
        self.embedding_dim = embedding_dim
        self.index = None
        self.n_vectors = 0

    def build(self, embeddings: np.ndarray, verbose: bool = True):
        """Build the index from embeddings

        Args:
            embeddings: Array of embeddings with shape (n_docs, embedding_dim)
            verbose: Whether to log build progress
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        self.n_vectors = len(embeddings)
        start_time = time.time()

        # Ensure embeddings are contiguous and float32
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))

        if self.config.index_type == "flat":
            self._build_flat_index(embeddings)
        elif self.config.index_type == "ivf":
            self._build_ivf_index(embeddings)
        else:
            raise ValueError(
                f"Unknown index type: {self.config.index_type}. "
                "Must be 'flat' or 'ivf'"
            )

        build_time = time.time() - start_time

        if verbose:
            logger.info(
                f"Built {self.config.index_type} index with {self.n_vectors} vectors "
                f"in {build_time:.2f}s"
            )

    def _build_flat_index(self, embeddings: np.ndarray):
        """Build flat (brute-force) index

        Args:
            embeddings: Embeddings array
        """
        # IndexFlatIP uses inner product (equivalent to cosine similarity if normalized)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

    def _build_ivf_index(self, embeddings: np.ndarray):
        """Build IVF (clustered) index

        Args:
            embeddings: Embeddings array
        """
        # Create quantizer (flat index for cluster centroids)
        quantizer = faiss.IndexFlatIP(self.embedding_dim)

        # Create IVF index
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            self.config.n_clusters,
            faiss.METRIC_INNER_PRODUCT
        )

        # Train the index (find cluster centroids)
        logger.info(f"Training IVF index with {self.config.n_clusters} clusters...")
        self.index.train(embeddings)

        # Add vectors to index
        self.index.add(embeddings)

        # Set search parameters
        self.index.nprobe = self.config.nprobe

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        distance_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors

        Args:
            query_embedding: Query vector (1D array)
            k: Number of results to return
            distance_threshold: Optional distance cutoff (filters results with distance < threshold)

        Returns:
            Tuple of (distances, indices)
            - distances: Similarity scores (higher is better for inner product)
            - indices: Indices of nearest neighbors
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Flatten results (since we only have one query)
        distances = distances[0]
        indices = indices[0]

        # Apply distance threshold if specified
        if distance_threshold is not None:
            mask = distances >= distance_threshold
            distances = distances[mask]
            indices = indices[mask]

        return distances, indices

    def save(self, save_path: str):
        """Save index to disk

        Args:
            save_path: Path to save index file
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_path))
        logger.info(f"Saved index to {save_path}")

    def load(self, load_path: str):
        """Load index from disk

        Args:
            load_path: Path to index file
        """
        self.index = faiss.read_index(str(load_path))
        self.n_vectors = self.index.ntotal

        # Set nprobe for IVF index
        if self.config.index_type == "ivf":
            self.index.nprobe = self.config.nprobe

        logger.info(f"Loaded index with {self.n_vectors} vectors from {load_path}")