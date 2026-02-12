"""ArXiv CS corpus data loader

This module handles loading and preprocessing the arXiv Computer Science papers.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
import arxiv
import time

logger = logging.getLogger(__name__)


class ArXivLoader:
    """Loader for arXiv CS corpus"""

    def __init__(self, data_dir: str, max_docs: Optional[int] = None):
        """Initialize the loader with data directory path

        Args:
            data_dir: Path to directory containing arXiv data
            max_docs: Maximum number of documents to load (None for all)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_docs = max_docs
        self.cache_file = self.data_dir / "arxiv_cs_corpus.json"

    def load_corpus(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Load the arXiv CS corpus

        Args:
            force_reload: If True, reload from source even if cache exists

        Returns:
            List of documents with metadata
        """
        if self.cache_file.exists() and not force_reload:
            logger.info(f"Loading cached corpus from {self.cache_file}")
            return self._load_from_cache()

        logger.info("Downloading arXiv papers using arXiv API...")
        documents = self._download_and_process()

        logger.info(f"Saving corpus to cache: {self.cache_file}")
        self._save_to_cache(documents)

        return documents

    def _download_and_process(self) -> List[Dict[str, Any]]:
        """Download and process arXiv papers using arXiv API

        Returns:
            List of processed documents
        """
        logger.info("Querying arXiv API for Computer Science papers...")

        # Define CS categories to search
        # Using OR logic to get papers from multiple CS subcategories
        cs_categories = [
            'cat:cs.AI',  # Artificial Intelligence
            'cat:cs.CL',  # Computation and Language
            'cat:cs.CV',  # Computer Vision
            'cat:cs.LG',  # Machine Learning
            'cat:cs.NE',  # Neural and Evolutionary Computing
        ]

        documents = []
        max_results_per_category = (self.max_docs // len(cs_categories)) if self.max_docs else 500

        for category in cs_categories:
            if self.max_docs and len(documents) >= self.max_docs:
                break

            logger.info(f"Fetching papers from {category}...")

            try:
                # Create search query
                search = arxiv.Search(
                    query=category,
                    max_results=max_results_per_category,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )

                # Fetch results
                for paper in search.results():
                    if self.max_docs and len(documents) >= self.max_docs:
                        break

                    # Extract relevant fields
                    doc = {
                        'doc_id': paper.entry_id.split('/')[-1],  # Extract arXiv ID
                        'title': paper.title,
                        'abstract': paper.summary,
                        'categories': ', '.join(paper.categories),
                        'authors': ', '.join([author.name for author in paper.authors]),
                        'published': str(paper.published),
                        'updated': str(paper.updated),
                    }

                    # Combine title and abstract as the main text
                    doc['text'] = f"{doc['title']}\n\n{doc['abstract']}"

                    documents.append(doc)

                    if len(documents) % 50 == 0:
                        logger.info(f"Collected {len(documents)} papers so far...")

            except Exception as e:
                logger.warning(f"Error fetching from {category}: {e}")
                continue

        logger.info(f"Successfully loaded {len(documents)} Computer Science papers")
        return documents

    def _load_from_cache(self) -> List[Dict[str, Any]]:
        """Load documents from cached JSON file

        Returns:
            List of documents
        """
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents from cache")
        return documents

    def _save_to_cache(self, documents: List[Dict[str, Any]]):
        """Save documents to cache file

        Args:
            documents: List of documents to save
        """
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} documents to cache")