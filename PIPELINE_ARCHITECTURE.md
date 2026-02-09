# RAG Pipeline Architecture

## Research Project: Empirical Study of Retrieval Design Choices

This document describes the complete pipeline architecture for studying how retrieval design choices affect RAG system performance.

---

## File Structure

```
retrieval-design-research/
├── src/                              # Source code
│   ├── data/                         # Stage 1-2: Data ingestion and preprocessing
│   │   ├── loader.py                 # ArXiv corpus loading
│   │   └── chunking.py               # Text chunking strategies
│   │
│   ├── retrieval/                    # Stage 3-5: Retrieval pipeline
│   │   ├── embeddings.py             # Embedding generation (fixed model)
│   │   ├── indexing.py               # Vector indexing (Flat/IVF)
│   │   └── retriever.py              # Retrieval orchestration
│   │
│   ├── generation/                   # Stage 6: Answer generation
│   │   └── llm.py                    # LLM-based generation (fixed model)
│   │
│   ├── evaluation/                   # Stage 8: Evaluation framework
│   │   ├── harness.py                # Experiment orchestration
│   │   └── metrics.py                # Evaluation metrics
│   │
│   ├── utils/                        # Utilities
│   │   └── config.py                 # Configuration management
│   │
│   └── rag_pipeline.py               # Stage 7: End-to-end pipeline
│
├── experiments/                      # Experiment configurations
│   ├── configs/
│   │   ├── baseline.yaml             # Baseline configuration
│   │   ├── experiment1_chunking.yaml # Chunking experiments
│   │   ├── experiment2_topk.yaml     # Retrieval parameter experiments
│   │   └── experiment3_indexing.yaml # Indexing strategy experiments
│   │
│   └── scripts/                      # Experiment runners
│       ├── run_baseline.py           # Setup baseline artifacts
│       ├── run_evaluation.py         # Run evaluation on config
│       ├── run_experiment.py         # Batch experiment runner
│       └── generate_questions.py     # Generate evaluation questions
│
├── data/                             # Data storage
│   ├── raw/                          # Raw downloaded data
│   │   └── arxiv_cs_corpus.json      # Cached arXiv papers
│   │
│   ├── processed/                    # Processed artifacts
│   │   ├── {config}_chunks.json      # Chunked documents
│   │   ├── {config}_embeddings.npy   # Generated embeddings
│   │   └── {config}_index.faiss      # FAISS vector index
│   │
│   └── questions/                    # Evaluation questions
│       └── eval_questions.json       # Question set with gold docs
│
├── results/                          # Experiment results
│   └── {config}_evaluation.csv       # Per-config evaluation results
│
├── notebooks/                        # Analysis notebooks (optional)
│
├── tests/                            # Unit tests
│
├── CLAUDE.md                         # Project instructions
├── README.md                         # Project overview
├── QUICKSTART.md                     # Quick start guide
├── requirements.txt                  # Python dependencies
└── .env                              # Environment variables (API keys)
```

---

## Complete Pipeline Flow

### **Stage 1: Data Ingestion**
**File:** `src/data/loader.py`
**Class:** `ArXivLoader`

```python
# Downloads arXiv Computer Science papers
loader = ArXivLoader(data_dir="data/raw", max_docs=500)
documents = loader.load_corpus()  # → List[Dict[str, Any]]
```

**Input:** arXiv API
**Output:** `data/raw/arxiv_cs_corpus.json`

**Document Schema:**
```json
{
  "doc_id": "2301.12345v1",
  "title": "Paper Title",
  "abstract": "Paper abstract...",
  "text": "Title\n\nAbstract...",
  "categories": "cs.LG, cs.AI",
  "authors": "Author1, Author2",
  "published": "2023-01-20",
  "updated": "2023-01-20"
}
```

**Categories Fetched:**
- `cs.AI` - Artificial Intelligence
- `cs.CL` - Computation and Language (NLP)
- `cs.CV` - Computer Vision
- `cs.LG` - Machine Learning
- `cs.NE` - Neural and Evolutionary Computing

---

### **Stage 2: Text Chunking**
**File:** `src/data/chunking.py`
**Class:** `TextChunker`

```python
# Chunk documents into smaller segments
config = ChunkConfig(chunk_size=512, overlap=0)
chunker = TextChunker(config)
chunks = chunker.chunk_documents(documents)  # → List[Dict[str, Any]]
```

**Input:** `documents` (from Stage 1)
**Output:** `data/processed/{config}_chunks.json`

**Chunk Schema:**
```json
{
  "chunk_id": "2301.12345v1_chunk_0",
  "doc_id": "2301.12345v1",
  "chunk_index": 0,
  "text": "Chunk text content...",
  "start_token": 0,
  "end_token": 512,
  "num_tokens": 512,
  "original_doc": {
    "doc_id": "...",
    "title": "...",
    "categories": "..."
  }
}
```

**Chunking Parameters (Experimental Variables):**
- `chunk_size`: 256 / 512 / 1024 tokens
- `overlap`: 0 / chunk_size//2 tokens
- Tokenizer: `tiktoken` (cl100k_base - OpenAI's tokenizer)

---

### **Stage 3: Embedding Generation**
**File:** `src/retrieval/embeddings.py`
**Class:** `EmbeddingModel`

```python
# Generate embeddings for all chunks
model = EmbeddingModel(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model.load()
embeddings = model.embed_texts(chunk_texts, normalize=True)  # → np.ndarray
```

**Input:** `chunks` (from Stage 2)
**Output:** `data/processed/{config}_embeddings.npy`

**Embedding Details:**
- **Model:** `multi-qa-MiniLM-L6-cos-v1` (FIXED across all experiments)
- **Dimension:** 384
- **Normalization:** Yes (for cosine similarity via inner product)
- **Batch Size:** 32
- **Output Shape:** `(n_chunks, 384)`

---

### **Stage 4: Vector Indexing**
**File:** `src/retrieval/indexing.py`
**Class:** `VectorIndex`

```python
# Build vector index for similarity search
config = IndexConfig(index_type="flat")
index = VectorIndex(config, embedding_dim=384)
index.build(embeddings)  # Builds FAISS index
```

**Input:** `embeddings` (from Stage 3)
**Output:** `data/processed/{config}_index.faiss`

**Index Types (Experimental Variable):**

1. **Flat Index** (Baseline):
   - `faiss.IndexFlatIP` (Inner Product)
   - Brute-force exact search
   - Slower, 100% accurate

2. **IVF Index** (Experiment):
   - `faiss.IndexIVFFlat` (Inverted File Index)
   - Clustered approximate search
   - Faster, ~98% accurate
   - Parameters: `n_clusters`, `nprobe`

---

### **Stage 5: Retrieval (Query Time)**
**File:** `src/retrieval/retriever.py`
**Class:** `Retriever`

```python
# Retrieve relevant chunks for a query
retriever = Retriever(embedding_model, vector_index, config, chunks)
result = retriever.retrieve(query="What are transformers?")  # → RetrievalResult
```

**Input:** Query string
**Output:** `RetrievalResult`

**Process:**
1. Embed query using `EmbeddingModel.embed_query()`
2. Search index: `vector_index.search(query_embedding, k=top_k)`
3. Fetch chunks by indices
4. Return documents + metadata

**RetrievalResult Schema:**
```python
{
  "query": "What are transformers?",
  "documents": [chunk1, chunk2, ...],  # Retrieved chunks
  "distances": [0.89, 0.85, ...],      # Similarity scores
  "chunk_ids": ["id1", "id2", ...],
  "latency_ms": 12.5,
  "num_results": 5
}
```

**Retrieval Parameters (Experimental Variables):**
- `top_k`: 3 / 5 / 10
- `distance_threshold`: None / custom cutoff

---

### **Stage 6: Answer Generation**
**File:** `src/generation/llm.py`
**Class:** `LLMGenerator`

```python
# Generate answer using LLM with retrieved context
generator = LLMGenerator(GenerationConfig(model_name="gpt-4o-mini"))
result = generator.generate_answer(query, context_documents)  # → GenerationResult
```

**Input:** Query + Retrieved documents (from Stage 5)
**Output:** `GenerationResult`

**Process:**
1. Build prompt with context:
   ```
   [Paper 1]
   Title: ...
   Categories: ...
   Content: {chunk_text}

   [Paper 2]
   ...

   Question: {query}
   ```
2. Call OpenAI API with system prompt
3. Return answer + metadata

**LLM Configuration (FIXED):**
- **Model:** `gpt-4o-mini`
- **Temperature:** 0.0 (deterministic)
- **Max Tokens:** 500
- **System Prompt:** CS/ML research assistant

**GenerationResult Schema:**
```python
{
  "query": "What are transformers?",
  "answer": "Transformers are neural network architectures...",
  "prompt": "Research Papers: [...] Question: ...",
  "model": "gpt-4o-mini",
  "latency_ms": 1250.0,
  "prompt_tokens": 856,
  "completion_tokens": 124,
  "total_tokens": 980
}
```

---

### **Stage 7: End-to-End RAG Pipeline**
**File:** `src/rag_pipeline.py`
**Class:** `RAGPipeline`

```python
# Complete RAG flow
pipeline = RAGPipeline(retriever, generator)
result = pipeline.query("What are transformers?")  # → RAGResult
```

**Process:**
```
Query → Retrieval (Stage 5) → Generation (Stage 6) → RAGResult
```

**RAGResult Schema:**
```python
{
  "query": "What are transformers?",
  "answer": "Transformers are...",
  "retrieval_result": RetrievalResult,
  "generation_result": GenerationResult,
  "total_latency_ms": 1262.5,
  "num_documents_used": 5
}
```

---

### **Stage 8: Evaluation**
**File:** `src/evaluation/harness.py`
**Class:** `EvaluationHarness`

```python
# Run evaluation on question set
harness = EvaluationHarness("data/questions/eval_questions.json")
harness.load_questions()
results = harness.run_evaluation(config_name, rag_pipeline)  # → List[EvaluationResult]
```

**Input:**
- Question set: `data/questions/eval_questions.json`
- RAG pipeline (Stage 7)

**Output:** `results/{config}_evaluation.csv`

**Question Schema:**
```json
{
  "question_id": "q001",
  "question": "What are attention mechanisms?",
  "gold_doc_ids": ["2301.12345v1", "2302.67890v1"],
  "answer": "Reference answer for evaluation"
}
```

**Evaluation Metrics:**

1. **Retrieval Recall:**
   - Do retrieved docs contain gold documents?
   - Formula: `|retrieved ∩ gold| / |gold|`

2. **Answer Correctness:**
   - LLM-judged correctness (0-2 scale)
   - 0 = incorrect, 1 = partial, 2 = correct

3. **Hallucination Detection:**
   - Does answer contradict context?
   - Binary: True/False

4. **Latency:**
   - Retrieval latency (ms)
   - Generation latency (ms)
   - Total latency (ms)

**EvaluationResult Schema:**
```csv
question_id,question,config_name,retrieved_doc_ids,retrieval_recall,answer,reference_answer,answer_correctness,has_hallucination,retrieval_latency_ms,generation_latency_ms,total_latency_ms
q001,"What are...",baseline,doc1|doc2|doc3,1.0,"Answer text","Reference",2.0,False,12.5,1250.0,1262.5
```

---

## Experimental Design

### **Fixed Variables (Control)**
- Dataset: arXiv CS papers (AI, CL, CV, LG, NE)
- Embedding model: `multi-qa-MiniLM-L6-cos-v1`
- LLM: `gpt-4o-mini`
- Evaluation questions: Same set for all experiments

### **Independent Variables (Vary)**

#### **Experiment 1: Chunking Strategy**
**Config:** `experiments/configs/experiment1_chunking.yaml`

| Variable | Values |
|----------|--------|
| `chunk_size` | 256, 512, 1024 tokens |
| `overlap` | 0, chunk_size//2 |

**Total Configs:** 6 (3 sizes × 2 overlap settings)

---

#### **Experiment 2: Retrieval Parameters**
**Config:** `experiments/configs/experiment2_topk.yaml`

| Variable | Values |
|----------|--------|
| `top_k` | 3, 5, 10 |
| `distance_threshold` | None, 0.5, 0.7 |

**Total Configs:** Variable

---

#### **Experiment 3: Indexing Strategy**
**Config:** `experiments/configs/experiment3_indexing.yaml`

| Variable | Values |
|----------|--------|
| `index_type` | flat, ivf |
| `n_clusters` | 50, 100, 200 (IVF only) |
| `nprobe` | 5, 10, 20 (IVF only) |

**Total Configs:** Variable

---

### **Dependent Variables (Measure)**
- Retrieval recall
- Answer correctness
- Hallucination rate
- Retrieval latency
- Generation latency
- Total latency

---

## Running Experiments

### **1. Setup Baseline**
```bash
# Download data, chunk, embed, index
python experiments/scripts/run_baseline.py \
  --config experiments/configs/baseline.yaml
```

**Creates:**
- `data/raw/arxiv_cs_corpus.json`
- `data/processed/baseline_chunks.json`
- `data/processed/baseline_embeddings.npy`
- `data/processed/baseline_index.faiss`

---

### **2. Generate Evaluation Questions**
```bash
python experiments/scripts/generate_questions.py \
  --corpus data/raw/arxiv_cs_corpus.json \
  --output data/questions/eval_questions.json \
  --num-questions 50
```

---

### **3. Run Baseline Evaluation**
```bash
python experiments/scripts/run_evaluation.py \
  --config-name baseline \
  --questions data/questions/eval_questions.json \
  --chunks data/processed/baseline_chunks.json \
  --index data/processed/baseline_index.faiss \
  --output results/baseline_evaluation.csv
```

---

### **4. Run Batch Experiments**
```bash
# Run all chunking experiments
python experiments/scripts/run_experiment.py \
  --base-config experiments/configs/baseline.yaml \
  --experiment-config experiments/configs/experiment1_chunking.yaml
```

---

## Data Flow Diagram

```
User Question
    ↓
┌─────────────────────────────────────────────────────┐
│ OFFLINE PROCESSING (Once per config)                │
├─────────────────────────────────────────────────────┤
│ ArXiv API                                            │
│   ↓ (loader.py)                                     │
│ Raw Documents (JSON)                                │
│   ↓ (chunking.py)                                   │
│ Chunks (JSON)                                       │
│   ↓ (embeddings.py)                                 │
│ Embeddings (NPY)                                    │
│   ↓ (indexing.py)                                   │
│ FAISS Index                                         │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ ONLINE PROCESSING (Per query)                       │
├─────────────────────────────────────────────────────┤
│ Query String                                         │
│   ↓ (retriever.py)                                  │
│ Query Embedding                                      │
│   ↓ (indexing.py)                                   │
│ Vector Search → Top-k Chunks                        │
│   ↓ (retriever.py)                                  │
│ Retrieved Documents + Scores                         │
│   ↓ (llm.py)                                        │
│ Build Prompt with Context                           │
│   ↓ (OpenAI API)                                    │
│ Generated Answer                                     │
│   ↓ (rag_pipeline.py)                               │
│ RAGResult (answer + metadata)                       │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ EVALUATION (Batch processing)                        │
├─────────────────────────────────────────────────────┤
│ For each question in eval set:                      │
│   - Run RAG pipeline                                │
│   - Calculate metrics                               │
│   - Compare to gold docs/answers                    │
│ → CSV with all results                              │
└─────────────────────────────────────────────────────┘
```

---

## Key Design Principles

1. **Reproducibility:** All configs in YAML, results in CSV
2. **Modularity:** Each stage is independent and testable
3. **Caching:** Artifacts saved to disk, reused when possible
4. **Controlled Experiments:** Only vary retrieval decisions, keep model/data fixed
5. **Research-Oriented:** Focus on systematic comparison, not production deployment

---

## Next Steps

1. ✅ Baseline setup complete
2. ✅ Evaluation questions generated
3. ✅ Baseline evaluation run
4. ⏳ Implement `run_experiment.py` for batch experiments
5. ⏳ Run Experiments 1-3
6. ⏳ Analyze results and failure cases
7. ⏳ Write research memo

---

## References

- **Embedding Model:** [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
- **Vector Search:** [FAISS](https://github.com/facebookresearch/faiss)
- **LLM:** [OpenAI GPT-4o-mini](https://platform.openai.com/docs/models)
- **Tokenizer:** [tiktoken](https://github.com/openai/tiktoken)
