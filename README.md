# Empirical Study of Retrieval Design Choices in RAG Systems

A research-focused project investigating how retrieval pipeline design choices influence answer quality, latency, and hallucination in Retrieval-Augmented Generation (RAG) systems.

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

## Results

### Exp 1 — Top-k Retrieval (5,000-doc corpus, 283 questions)

| Config | Recall | Miss% | Correctness | Hallucination | Abstention | Total latency |
|---|---|---|---|---|---|---|
| top_k=1  | 0.676 | 32.4% | 1.409 | 6.7% | 1.552 | 3,818ms |
| top_k=3  | 0.760 | 24.0% | 1.564 | 3.9% | 1.069 | 3,959ms |
| top_k=5  | 0.791 | 20.9% | 1.653 | 2.8% | 1.086 | 3,795ms |
| top_k=10 | 0.827 | 17.3% | 1.684 | 4.6% | 1.155 | 4,217ms |
| top_k=20 | 0.853 | 14.7% | 1.720 | 3.9% | 1.172 | 6,348ms |

*Recall and Correctness are for answerable questions only. Abstention is for unanswerable questions (max 2.0). Miss% = fraction of answerable questions where the gold doc was not retrieved.*

**Findings:**
- Retrieval recall improves with higher k but with diminishing returns (+8.4pp, +3.1pp, +3.6pp, +2.6pp per step)
- Answer correctness tracks recall tightly — generation quality is high; retrieval is the bottleneck
- Hallucination is worst at top_k=1 (6.7%) and lowest at top_k=5 (2.8%); adding more context beyond k=5 slightly increases hallucination
- Abstention degrades above top_k=1 — more retrieved context makes the model more likely to attempt an answer on unanswerable questions
- Total latency is flat up to top_k=10, then jumps 67% at top_k=20 due to larger context windows

### Exp 2 — Similarity Threshold (distance_threshold)

| Config | Recall (ans.) | Correctness (ans.) | Hallucination | Abstention (unans.) | Avg Docs | Total latency |
|---|---|---|---|---|---|---|
| threshold=none | 0.827 | 1.689 | 4.2% | 1.155 | 8.2 | 4,785ms |
| threshold=0.3  | 0.822 | 1.680 | 4.9% | 1.172 | 7.8 | 3,552ms |
| threshold=0.5  | 0.693 | 1.404 | 17.7% | 1.569 | 2.4 | 2,930ms |
| threshold=0.7  | 0.147 | 0.347 | 66.4% | 1.741 | 0.1 | 1,845ms |

*Correctness on 0–2 scale. Abstention max 2.0. 80 questions retrieved zero docs at threshold=0.5; 249/283 at threshold=0.7.*

**Correctness distribution (answerable questions):**

| Config | Score=0 (wrong) | Score=1 (partial) | Score=2 (correct) |
|---|---|---|---|
| threshold=none | 11.6% | 8.0% | 80.4% |
| threshold=0.3  | 12.4% | 7.1% | 80.4% |
| threshold=0.5  | 27.1% | 5.3% | 67.6% |
| threshold=0.7  | 80.0% | 5.3% | 14.7% |

**Findings:**
- `none` and `0.3` are nearly equivalent in quality — the lenient threshold barely filters (8.2→7.8 avg docs) — but `0.3` cuts ~1.2s of latency with no meaningful quality penalty
- `0.5` is the inflection point: recall drops to 0.693, hallucinations spike 4× (4.2%→17.7%), and 80 questions receive zero retrieved documents
- `0.7` is effectively no retrieval — 249/283 questions retrieve zero docs, correctness collapses to 14.7% full-credit, and hallucination rate hits 66.4%
- Abstention on unanswerable questions improves with stricter thresholds (model correctly declines more often when context is empty), but this is entirely offset by the collapse on answerable questions
- Retrieval quality is the binding constraint: when relevant docs are filtered out, the LLM hallucinates rather than abstains

Exp 3 (indexing strategies) pending.

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
│   ├── compare_results.py        # Print summary table from result CSVs
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

### Step 4 — Compare results

```bash
python scripts/compare_results.py                        # exp1 top-k (default)
python scripts/compare_results.py --pattern "exp2_*"    # exp2
python scripts/compare_results.py --pattern "exp3_*"    # exp3
```

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

## References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [arXiv API](https://arxiv.org/help/api)
