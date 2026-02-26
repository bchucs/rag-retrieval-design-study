# Empirical Analysis of Retrieval Design Choices in Retrieval-Augmented Generation Systems

A research-focused project investigating how retrieval pipeline design choices influence answer quality, latency, and hallucination in Retrieval-Augmented Generation (RAG) systems.

## Research Question

> How do retrieval pipeline design choices (top-k selection, distance thresholds, indexing methods) affect answer quality, latency, and hallucination rates in RAG systems?

## Overview

Controlled experiments isolate the impact of retrieval parameters by fixing the dataset, embedding model, and LLM across all runs.

### Fixed Components
| Component | Value |
|---|---|
| Corpus | arXiv CS papers (cs.AI, cs.CL, cs.CV, cs.LG, cs.NE) — 50,000 documents → 32,304 chunks |
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

### Exp 1 — Top-k Retrieval (50,000-doc corpus, 283 questions)

| Config | Recall | Miss% | Correctness | Hallucination | Abstention | Median latency |
|---|---|---|---|---|---|---|
| top_k=1  | 0.498 | 50.2% | 1.160 | 13.8% | 1.603 | 3,303ms |
| top_k=3  | 0.578 | 42.2% | 1.347 | 11.1% | 1.086 | 3,861ms |
| top_k=5  | 0.631 | 36.9% | 1.427 | 10.2% | 1.155 | 3,995ms |
| top_k=10 | 0.662 | 33.8% | 1.502 | 10.7% | 1.103 | 3,894ms |
| top_k=20 | 0.711 | 28.9% | 1.596 | 9.3%  | 1.086 | 6,233ms |

*Recall and Correctness are for answerable questions only. Abstention is for unanswerable questions (max 2.0). Miss% = fraction of answerable questions where the gold doc was not retrieved. Latency is median across all questions.*

**Findings:**
- Recall is substantially lower across the board vs the 5k corpus — at 10× scale, top_k=20 (0.711) barely matches what top_k=3 achieved at 5k (0.760), confirming that optimal k must be higher at larger corpus sizes
- Miss rate remains high even at top_k=20 (28.9%), suggesting the evaluation set's gold docs are genuinely harder to surface in a 50k document pool
- Hallucination is worst at top_k=1 (13.8%) and improves monotonically with k, reaching 9.3% at top_k=20 — unlike the 5k run, there is no U-shaped pattern; more context continues to help throughout this range
- Abstention on unanswerable questions is best at top_k=1 (1.603) and degrades with more context, consistent with the 5k findings
- Latency is flat up to top_k=10 (3.3–4.0s), then jumps ~56% at top_k=20 due to larger context windows sent to the LLM

### Exp 2 — Similarity Threshold (distance_threshold)

| Config | Recall (ans.) | Correctness (ans.) | Hallucination | Abstention (unans.) | Avg Docs | Median latency |
|---|---|---|---|---|---|---|
| threshold=none | 0.662 | 1.511 | 9.8%  | 1.121 | 8.3 | 4,763ms |
| threshold=0.3  | 0.662 | 1.507 | 9.3%  | 1.121 | 8.3 | 4,193ms |
| threshold=0.5  | 0.596 | 1.364 | 21.3% | 1.345 | 4.6 | 3,844ms |
| threshold=0.7  | 0.133 | 0.360 | 81.3% | 1.707 | 0.2 | 1,449ms |

*Correctness on 0–2 scale. Abstention max 2.0. 51 questions retrieved zero docs at threshold=0.5; 244/283 at threshold=0.7.*

**Correctness distribution (answerable questions):**

| Config | Score=0 (wrong) | Score=1 (partial) | Score=2 (correct) |
|---|---|---|---|
| threshold=none | 19.1% | 10.7% | 70.2% |
| threshold=0.3  | 20.0% | 9.3%  | 70.7% |
| threshold=0.5  | 27.1% | 9.3%  | 63.6% |
| threshold=0.7  | 80.0% | 4.0%  | 16.0% |

**Findings:**
- `none` and `0.3` remain nearly equivalent in quality (both 0.662 recall, 70%+ full-credit answers) — the threshold barely filters at this corpus density (8.3 avg docs in both cases) — but `0.3` saves ~570ms of median latency
- At 50k docs the full-credit rate drops to ~70% vs 80% at 5k, reflecting the lower baseline recall; the relative pattern across thresholds is preserved
- `0.5` is again the inflection point: recall drops to 0.596, hallucinations more than double (9.8%→21.3%), and 51 questions receive zero retrieved documents
- `0.7` is effectively no retrieval — 244/283 questions retrieve zero docs, correctness collapses to 16.0% full-credit, and hallucination hits 81.3% (worse than the 66.4% seen at 5k scale)
- The 0.7 threshold becomes more catastrophic at 50k: with more documents, similarity scores compress and fewer docs clear the high bar, making the cliff steeper
- Retrieval quality remains the binding constraint: when relevant docs are filtered out, the LLM hallucinates rather than abstains

### Exp 3 — Index Strategy

IVF-conservative and IVF-balanced use `n_clusters=100`; IVF-aggressive uses `n_clusters=200`. All vary `nprobe` (number of clusters searched at query time).

| Config | n_clusters | nprobe | Recall (ans.) | Correctness (ans.) | Hallucination | Abstention (unans.) | Ret. latency | Median latency |
|---|---|---|---|---|---|---|---|---|
| flat             | —   | —  | 0.631 | 1.422 | 10.2% | 1.000 | 58ms | 3,505ms |
| ivf_conservative | 100 | 20 | 0.542 | 1.267 | 16.4% | 1.121 | 52ms | 3,937ms |
| ivf_balanced     | 100 | 10 | 0.453 | 1.089 | 24.9% | 1.155 | 47ms | 3,324ms |
| ivf_aggressive   | 200 | 5  | 0.360 | 0.871 | 32.9% | 1.121 | 61ms | 3,789ms |

**Correctness distribution (answerable questions):**

| Config | Score=0 (wrong) | Score=1 (partial) | Score=2 (correct) |
|---|---|---|---|
| flat             | 23.6% | 10.7% | 65.8% |
| ivf_conservative | 31.1% | 11.1% | 57.8% |
| ivf_balanced     | 39.1% | 12.9% | 48.0% |
| ivf_aggressive   | 51.1% | 10.7% | 38.2% |

**Findings:**
- Retrieval latency is again nearly identical across configs (47–61ms) — even at 50k docs, retrieval is so fast that IVF provides no measurable latency advantage over flat brute-force
- Accuracy degrades severely and monotonically with lower nprobe — the accuracy cost of IVF is much larger at 50k than at 5k, as the larger corpus amplifies approximation errors
- ivf_conservative (nprobe=20) costs 8.9pp recall and 6.4pp full-credit answers vs flat, with zero latency benefit — a worse trade than at 5k scale
- ivf_aggressive drops full-credit answers to 38.2% (vs 65.8% for flat), with hallucination tripling to 32.9% — an unacceptable quality loss for sub-millisecond savings
- The case for IVF remains nil at this scale: end-to-end latency is dominated by LLM generation (~3.3–3.9s), making retrieval differences of <15ms irrelevant; flat indexing is the clear choice
- The scale-up hypothesis — that IVF would show real latency benefits at 50k — did not materialize; the crossover point is likely at a much larger corpus (hundreds of thousands to millions of docs)

---

## Scale-up Results (5k → 50k)

All three experiments were repeated on a 50,000-document corpus (32,304 chunks) to test whether findings held or reversed at scale. Results above reflect the 50k run.

| Hypothesis | Outcome |
|---|---|
| Exp 1: optimal k shifts higher at scale | **Confirmed** — recall at top_k=20 (0.711) barely matches top_k=3 at 5k (0.760); miss rates remain 29–50% throughout |
| Exp 2: threshold inflection point may shift | **Partially confirmed** — 0.5 threshold is slightly less catastrophic (51 vs 80 zero-doc questions), but hallucination still doubles; 0.7 becomes *worse* |
| Exp 3: IVF latency advantage emerges at scale | **Not confirmed** — retrieval latency is still dominated by LLM generation at 50k; IVF accuracy loss is larger at scale with no speed benefit |

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
