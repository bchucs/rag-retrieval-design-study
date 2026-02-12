"""Embedding generation using fixed model

Uses a single embedding model throughout all experiments.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for embedding model"""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """Initialize embedding model

        Args:
            model_name: Name of the embedding model to use
            batch_size: Batch size for encoding
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = None
        self._embedding_dim = None

    def load(self):
        """Load the embedding model"""
        logger.info(f"Loading embedding model: {self.model_name}")

        cache_folder = str(self.cache_dir) if self.cache_dir else None

        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=cache_folder
        )

        # Get embedding dimension
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Model loaded. Embedding dimension: {self._embedding_dim}"
        )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension

        Returns:
            Dimension of embedding vectors
        """
        if self._embedding_dim is None:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        return self._embedding_dim

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings (recommended for cosine similarity)

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not texts:
            return np.array([])

        # Filter out empty texts
        valid_texts = [t if t and t.strip() else " " for t in texts]

        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        return embeddings

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single query

        Args:
            query: Query string
            normalize: Whether to normalize embedding

        Returns:
            Query embedding vector
        """
        return self.embed_texts([query], normalize=normalize)[0]

    def save_embeddings(self, embeddings: np.ndarray, save_path: str):
        """Save embeddings to disk

        Args:
            embeddings: Embeddings array
            save_path: Path to save file (.npy)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(save_path, embeddings)
        logger.info(f"Saved {len(embeddings)} embeddings to {save_path}")

    def load_embeddings(self, load_path: str) -> np.ndarray:
        """Load embeddings from disk

        Args:
            load_path: Path to embeddings file (.npy)

        Returns:
            Loaded embeddings array
        """
        embeddings = np.load(load_path)
        logger.info(f"Loaded {len(embeddings)} embeddings from {load_path}")
        return embeddings