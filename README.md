# Empirical Study of Retrieval Design Choices in RAG Systems

A research-focused project studying how retrieval pipeline design choices affect answer quality, latency, and hallucination rates in Retrieval-Augmented Generation (RAG) systems.

## Research Question

> How do retrieval pipeline design choices (top-k selection, distance thresholds, indexing methods) affect answer quality, latency, and hallucination rates in RAG systems?

## Overview

Controlled experiments isolate the impact of retrieval parameters by fixing the dataset, embedding model, and LLM across all runs.

### Fixed Components
| Component | Value |
|---|---|
| Corpus | arXiv CS papers (cs.AI, cs.CL, cs.CV, cs.LG, cs.NE) — 5,000 documents |
| Chunking | 512 tokens, no overlap, `cl100k_base` tokenizer |
| Embedding model | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (384-dim) |
| LLM | `gpt-4o-mini`, temperature=0.0, max_tokens=500 |
| Evaluation set | 283 questions (225 answerable, 58 unanswerable) |

### Experiments
| | Variable | Configs |
|---|---|---|
| **Exp 1** | Retrieval quantity (`top_k`) | 1, 3, 5, 10, 20 |
| **Exp 2** | Similarity filtering (`distance_threshold`) | none, 0.3, 0.5, 0.7 |
| **Exp 3** | Index type | flat, IVF-conservative, IVF-balanced, IVF-aggressive |

---

## Project Structure

```
.
├── src/                          # Source modules
│   ├── rag_pipeline.py           # End-to-end RAG orchestration
│   ├── data/
│   │   ├── loader.py             # ArXiv corpus loader (API + caching)
│   │   └── chunking.py           # Token-based text chunking
│   ├── retrieval/
│   │   ├── embeddings.py         # SentenceTransformers wrapper
│   │   ├── indexing.py           # FAISS flat/IVF index builder
│   │   └── retriever.py          # Query embedding + vector search
│   ├── generation/
│   │   └── llm.py                # OpenAI GPT interface
│   ├── evaluation/
│   │   ├── metrics.py            # Recall, correctness, hallucination, abstention
│   │   └── harness.py            # Evaluation orchestration + CSV export
│   └── utils/
│       └── config.py             # YAML config loading + factory functions
│
├── experiments/
│   ├── configs/
│   │   ├── baseline.yaml         # Reference config (top_k=5, flat index)
│   │   ├── template.yaml         # Documented template with all options
│   │   ├── exp1/                 # top_k variants (1, 3, 5, 10, 20)
│   │   ├── exp2/                 # distance_threshold variants
│   │   └── exp3/                 # indexing strategy variants
│   └── scripts/
│       └── run_evaluation.py     # Evaluation runner for a single config
│
├── scripts/
│   ├── build_retrieval_system.py # Build chunks, embeddings, and FAISS index
│   ├── generate_questions.py     # LLM-based eval question generation
│   └── run_all_experiments.sh    # Batch runner for all configs
│
├── data/
│   ├── raw/
│   │   └── arxiv_cs_corpus.json  # Cached arXiv corpus
│   ├── processed/
│   │   ├── baseline_chunks.json       # Chunked documents
│   │   ├── baseline_metadata.json     # Artifact metadata + hash
│   │   ├── baseline_embeddings.npy    # Precomputed embeddings
│   │   └── baseline_index.faiss       # FAISS flat index
│   └── questions/
│       └── eval_questions.json        # 283 evaluation questions
│
├── results/                      # Experiment output CSVs
│   └── logs/                     # Per-experiment run logs
└── notebooks/                    # Analysis notebooks
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Workflow

### Step 1 — Build retrieval artifacts

Downloads the arXiv corpus, chunks documents, generates embeddings, and builds the FAISS index. Artifacts are cached and reused across experiments that share the same chunking/embedding config.

```bash
python scripts/build_retrieval_system.py
```

Outputs to `data/processed/`: `baseline_chunks.json`, `baseline_embeddings.npy`, `baseline_index.faiss`, `baseline_metadata.json`.

### Step 2 — Generate evaluation questions

Uses GPT to generate factual questions from document abstracts. Produces answerable questions (with gold doc IDs and reference answers) and unanswerable questions.

```bash
python scripts/generate_questions.py
```

Outputs to `data/questions/eval_questions.json`.

### Step 3 — Run experiments

**Single config:**
```bash
python experiments/scripts/run_evaluation.py \
    --config experiments/configs/exp1/exp1_topk_5.yaml
```

**All 13 configs in batch:**
```bash
./scripts/run_all_experiments.sh
```

**Options:**
```bash
# Limit questions (useful for smoke testing)
./scripts/run_all_experiments.sh --num-questions 20

# Run only one experiment group
./scripts/run_all_experiments.sh --configs-dir experiments/configs/exp1

# Preview commands without executing
./scripts/run_all_experiments.sh --dry-run
```

Results are written to `results/<exp_name>_evaluation_<n>.csv`. Logs go to `results/logs/<exp_name>.log`.

> **Note**: Exp 3 configs use experiment-specific FAISS indices (`{exp_name}_index.faiss`). Build these before running exp3: `python scripts/build_retrieval_system.py --config experiments/configs/exp3/<config>.yaml`

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `retrieval_recall` | Fraction of gold documents retrieved in top-k |
| `answer_correctness` | LLM-judged answer quality (0–2 scale) |
| `has_hallucination` | Whether the answer contains unsupported claims |
| `abstention_score` | For unanswerable questions: whether the model correctly declines (0 or 2) |
| `retrieval_latency_ms` | Time to embed query and search index |
| `generation_latency_ms` | Time to generate answer via LLM |
| `total_latency_ms` | End-to-end latency |

### Result CSV columns
`question_id`, `question`, `config_name`, `answerable`, `retrieved_doc_ids`, `retrieval_recall`, `answer`, `reference_answer`, `answer_correctness`, `has_hallucination`, `abstention_score`, `retrieval_latency_ms`, `generation_latency_ms`, `total_latency_ms`

---

## Results
Ongoing

## References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [arXiv API](https://arxiv.org/help/api)
