"""Text chunking strategies for retrieval

Implements various chunking approaches with configurable parameters.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking

    Note: Default values are provided, but load from experiments/configs/baseline.yaml in practice
    """
    chunk_size: int = 512  # tokens - override with config
    overlap: int = 0  # tokens - override with config
    encoding_name: str = "cl100k_base"  # OpenAI's tokenizer


class TextChunker:
    """Chunks text into smaller segments for embedding"""

    def __init__(self, config: ChunkConfig):
        """Initialize chunker with configuration

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.encoding = tiktoken.get_encoding(config.encoding_name)

    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Split text into chunks

        Args:
            text: Input text to chunk
            doc_id: Document identifier

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []

        # Encode text to tokens
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        chunks = []
        chunk_idx = 0
        start_idx = 0

        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.config.chunk_size, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Create chunk metadata
            chunk = {
                'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                'doc_id': doc_id,
                'chunk_index': chunk_idx,
                'text': chunk_text,
                'start_token': start_idx,
                'end_token': end_idx,
                'num_tokens': len(chunk_tokens),
            }

            chunks.append(chunk)

            # Move to next chunk with overlap
            step_size = self.config.chunk_size - self.config.overlap
            start_idx += step_size
            chunk_idx += 1

            # Prevent infinite loop if overlap >= chunk_size
            if step_size <= 0:
                logger.warning(
                    f"Overlap ({self.config.overlap}) >= chunk_size "
                    f"({self.config.chunk_size}), forcing step_size=1"
                )
                start_idx = end_idx

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'text'
    ) -> List[Dict[str, Any]]:
        """Chunk multiple documents

        Args:
            documents: List of documents to chunk
            text_field: Field name containing text to chunk

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            doc_id = doc.get('doc_id', 'unknown')
            text = doc.get(text_field, '')

            if not text:
                logger.warning(f"Document {doc_id} has no text in field '{text_field}'")
                continue

            chunks = self.chunk_text(text, doc_id)

            # Add original document metadata to each chunk
            for chunk in chunks:
                chunk['original_doc'] = {
                    k: v for k, v in doc.items()
                    if k != text_field  # Don't duplicate the full text
                }

            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)/len(documents):.1f} chunks/doc)"
        )

        return all_chunks