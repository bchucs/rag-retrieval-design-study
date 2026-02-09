# Quick Start Guide

## Step 1: Setup Environment

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables (optional for Step 1)

Create a `.env` file for API keys (needed for Step 2):

```bash
OPENAI_API_KEY=your_key_here
```

## Step 1: Baseline Retrieval Pipeline

### Run with Small Dataset (Recommended for Testing)

```bash
python experiments/scripts/run_baseline.py --max-docs 100
```

This will:
1. Download 100 arXiv CS papers from HuggingFace
2. Chunk them into 512-token pieces
3. Generate embeddings using `all-MiniLM-L6-v2`
4. Build a FAISS flat index
5. Test retrieval with sample queries

**Expected time**: 2-5 minutes

**Expected output**:
```
Loaded 100 documents
Created ~150-200 chunks
Generated embeddings
Built flat index
Sample retrieval results with latencies
```

### Run with Full Dataset

```bash
python experiments/scripts/run_baseline.py
```

This processes the entire arXiv CS corpus. **Warning**: This may take 30-60 minutes and require significant disk space.

### Cached Artifacts

After running, artifacts are saved to `data/processed/`:
- `baseline_chunks.json` - All document chunks with metadata
- `baseline_embeddings.npy` - Embedding vectors
- `baseline_index.faiss` - FAISS index

Re-running the script will reuse cached data if available.

## Testing Retrieval

### Interactive Testing

Create a test script `test_retrieval.py`:

```python
import sys
sys.path.insert(0, "src")

import json
import numpy as np
from pathlib import Path
from retrieval.embeddings import EmbeddingModel
from retrieval.indexing import VectorIndex, IndexConfig
from retrieval.retriever import Retriever, RetrievalConfig

# Load artifacts
processed_dir = Path("data/processed")

with open(processed_dir / "baseline_chunks.json") as f:
    chunks = json.load(f)

# Load embedding model
embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
embedding_model.load()

# Load index
index_config = IndexConfig(index_type="flat")
vector_index = VectorIndex(index_config, embedding_model.embedding_dim)
vector_index.load(str(processed_dir / "baseline_index.faiss"))

# Create retriever
retrieval_config = RetrievalConfig(top_k=5)
retriever = Retriever(embedding_model, vector_index, retrieval_config, chunks)

# Test query
query = "transformer architecture attention mechanism"
result = retriever.retrieve(query)

print(f"\nQuery: {query}")
print(f"Found {result.num_results} results in {result.latency_ms:.2f}ms\n")

for i, (doc, score) in enumerate(zip(result.documents, result.distances)):
    print(f"{i+1}. Score: {score:.3f}")
    print(f"   Title: {doc['original_doc']['title']}")
    print(f"   Chunk: {doc['text'][:200]}...")
    print()
```

Run:
```bash
python test_retrieval.py
```

## Troubleshooting

### Issue: Dataset download fails

**Solution**: The HuggingFace `arxiv_dataset` might not be available. Alternative:

1. Download from Kaggle: https://www.kaggle.com/Cornell-University/arxiv
2. Place in `data/raw/`
3. Update `loader.py` to read from local files

### Issue: Out of memory

**Solution**: Use `--max-docs` to limit dataset size:
```bash
python experiments/scripts/run_baseline.py --max-docs 50
```

### Issue: FAISS installation fails

**Solution**:
```bash
# Try CPU version specifically
pip install faiss-cpu --no-cache-dir

# Or build from source
conda install -c conda-forge faiss-cpu
```

## Next Steps

After Step 1 is complete:

1. **Step 2**: Implement end-to-end RAG pipeline with LLM generation
2. **Step 3**: Build evaluation harness with question set
3. **Step 4-6**: Run experiments varying chunking, top-k, and indexing

See CLAUDE.md for the full research plan.
