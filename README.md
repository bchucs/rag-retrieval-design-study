# Empirical Study of Retrieval Design Choices in RAG Systems

A research-focused project studying how retrieval pipeline design choices affect answer quality, latency, and hallucination rates in Retrieval-Augmented Generation (RAG) systems.

## Research Question

> How do retrieval pipeline design choices (chunking strategies, top-k selection, indexing methods) affect answer quality, latency, and hallucination rates in RAG systems?

## Project Overview

This project conducts **controlled experiments** to understand the tradeoffs between different retrieval design decisions in RAG systems. By fixing the dataset, embedding model, and LLM, we isolate the impact of retrieval parameters.

### Fixed Components
- **Dataset**: arXiv Computer Science papers/abstracts
- **Embedding Model**: sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (optimized for Q&A)
- **LLM**: GPT-4o-mini

**Note**: All model configurations are centralized in `experiments/configs/baseline.yaml`. Update this file to change models across all scripts.

### Variable Components (Experiments)
1. **Chunking strategies**: chunk size (256/512/1024 tokens), overlap vs no overlap
2. **Top-k selection**: top-k values (3/5/10), distance thresholds
3. **Indexing methods**: flat index, IVF clustering

## Project Structure

```
.
├── src/                       # Source code
│   ├── data/                  # Data loading and chunking
│   ├── retrieval/             # Embedding, indexing, retrieval
│   ├── generation/            # LLM interface
│   ├── evaluation/            # Metrics and evaluation harness
│   └── utils/                 # Logging and utilities
├── experiments/               # Experiment configurations and scripts
│   ├── configs/               # YAML configuration files
│   └── scripts/               # Experiment runner scripts
├── data/                      # Data storage
│   ├── raw/                   # Original arXiv data
│   ├── processed/             # Processed embeddings and indexes
│   └── questions/             # Evaluation question sets
├── results/                   # Experiment results (CSV)
├── notebooks/                 # Analysis notebooks
└── tests/                     # Unit tests
```

## Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Download Data

```bash
# TODO: Add data download instructions
# For now, implement data loader in src/data/loader.py
```

## Running Experiments

### Baseline Experiment

```bash
python experiments/scripts/run_baseline.py --config experiments/configs/baseline.yaml
```

### Experiment 1: Chunking Strategies

```bash
python experiments/scripts/run_experiment.py \
    --base-config experiments/configs/baseline.yaml \
    --experiment-config experiments/configs/experiment1_chunking.yaml
```

### Experiment 2: Top-k and Distance Thresholds

```bash
python experiments/scripts/run_experiment.py \
    --base-config experiments/configs/baseline.yaml \
    --experiment-config experiments/configs/experiment2_topk.yaml
```

### Experiment 3: Indexing Strategies

```bash
python experiments/scripts/run_experiment.py \
    --base-config experiments/configs/baseline.yaml \
    --experiment-config experiments/configs/experiment3_indexing.yaml
```

## Metrics

The evaluation harness measures:

- **Retrieval Recall**: Fraction of gold documents retrieved in top-k
- **Answer Correctness**: 0-2 scale evaluation of answer quality
- **Hallucination Rate**: Percentage of answers containing unsupported claims
- **Latency**: Retrieval and generation time (milliseconds)

## Results

Results are saved as CSV files in the `results/` directory with the following columns:
- question_id
- config_name
- retrieved_doc_ids
- retrieval_recall
- answer
- answer_correctness
- has_hallucination
- retrieval_latency_ms
- generation_latency_ms
- total_latency_ms

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ experiments/
flake8 src/ experiments/
```

## Implementation Roadmap

- [ ] **Step 1**: Dataset loader and baseline retrieval
- [ ] **Step 2**: End-to-end RAG pipeline
- [ ] **Step 3**: Evaluation harness and metrics
- [ ] **Step 4**: Experiment 1 - Chunking strategies
- [ ] **Step 5**: Experiment 2 - Top-k and thresholds
- [ ] **Step 6**: Experiment 3 - Indexing strategies
- [ ] **Step 7**: Failure analysis
- [ ] **Step 8**: Research writeup
- [ ] **Step 9**: Repository cleanup

## Key Findings

_To be filled after experiments are complete_

## Limitations

_To be filled after experiments are complete_

## Future Work

_To be filled after experiments are complete_

## License

MIT License

## References

- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- arXiv Dataset: https://www.kaggle.com/Cornell-University/arxiv