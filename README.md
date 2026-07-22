# PubMed QA Assistant

[![CI](https://github.com/arsalan9702/pubmedqa-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/arsalan9702/pubmedqa-rag/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) system for answering biomedical research questions, built on the PubMedQA dataset. Combines hybrid retrieval (dense + BM25), a custom fine-tuned BioBERT reranker, and grounded LLM generation to produce cited yes/no/maybe answers with supporting evidence.

---

## Overview

Given a research question, the system:

1. Retrieves candidate PubMed abstracts using hybrid dense + sparse search
2. Reranks candidates with a custom-trained BioBERT cross-encoder
3. Generates a grounded answer (yes / no / maybe + explanation) citing the retrieved evidence

Built as an end-to-end exploration of the modern RAG stack — not a wrapper around a single API call. The reranker is trained from scratch on domain data, and every stage is independently evaluated.

## Why this exists

Most RAG demos stop at "plug a vector DB into an LLM." This project asks a more specific question: **does retrieval quality actually change answer accuracy, and by how much?** The evaluation section below exists to answer that with numbers, not just a working demo.

## Architecture

```
                    ┌─────────────────┐
   question   ────▶ │  Hybrid Retriever │
                    │  (Dense + BM25)   │
                    └────────┬─────────┘
                             │ top-k candidates
                             ▼
                    ┌─────────────────┐
                    │  BioBERT Reranker │
                    │  (custom trained) │
                    └────────┬─────────┘
                             │ top-k reranked
                             ▼
                    ┌─────────────────┐
                    │   LLM Generator   │
                    │ (grounded prompt) │
                    └────────┬─────────┘
                             │
                             ▼
                answer (yes/no/maybe)
                + explanation + citations
```

## Dataset

[PubMedQA](https://pubmedqa.github.io/) — biomedical yes/no/maybe question answering over PubMed abstracts.

| Split | Size | Use |
|---|---|---|
| PQA-Labeled | 1,000 | Gold evaluation set (expert-annotated) |
| PQA-Unlabeled | 61.2k | Additional retrieval corpus |
| PQA-Artificial | 211.3k | Reranker pretraining signal |

## Project structure

```
pubmedqa-rag/
├── .github/
│   └── workflows/
│       └── ci.yml            # lint + test on every push/PR
├── data/
│   ├── raw/                  # cached HF dataset downloads
│   └── processed/            # cleaned chunks + QA pairs
├── src/
│   ├── preprocessing/        # data loading, chunking, indexing
│   ├── retrieval/            # dense, BM25, hybrid retrievers
│   ├── reranker/              # BioBERT cross-encoder: train + inference
│   ├── generation/           # LLM answer generation
│   └── evaluation/            # retrieval + answer metrics, ablations
├── tests/                     # pytest suite
├── notebooks/                 # Colab experiments (reranker training)
├── app/
│   └── streamlit_app.py      # demo UI
├── configs/
│   └── config.yaml           # single source of truth for all settings
├── pyproject.toml            # dependencies, ruff/black/pytest config
└── uv.lock                   # locked dependency versions
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

```fish
git clone git@github.com:arsalan9702/pubmedqa-rag.git
cd pubmedqa-rag

uv sync
source .venv/bin/activate.fish

cp .env.example .env    # add your free-tier LLM API key
```

Load the dataset:

```fish
uv run python src/preprocessing/load_data.py
```

Build the index:

```fish
uv run python src/preprocessing/build_index.py
```

Run the app:

```fish
uv run streamlit run app/streamlit_app.py
```

## Development

Lint and format:

```fish
uv run ruff check .
uv run black .
```

Run tests:

```fish
uv run pytest -v
```

All of the above run automatically on every push and pull request via GitHub Actions (see `.github/workflows/ci.yml`).

## Compute

| Stage | Where | Cost |
|---|---|---|
| Data loading, chunking, indexing | Local (CPU) | Free |
| Dense + BM25 retrieval | Local (CPU) | Free |
| Reranker fine-tuning | Colab (free T4 GPU) | Free |
| Answer generation | Free-tier LLM API (or local small model) | Free |

No paid infrastructure required at any stage.

## Evaluation

Three questions this project answers empirically:

1. **Does retrieval help at all?** — LLM-only (no context) vs. RAG accuracy
2. **Does hybrid search beat single-method retrieval?** — BM25-only vs. dense-only vs. hybrid, measured by recall@k
3. **Does the custom reranker earn its place?** — hybrid retrieval vs. hybrid + reranker, measured by downstream answer accuracy

| Metric | Description |
|---|---|
| Recall@k | Fraction of gold contexts retrieved in top-k |
| MRR | Mean reciprocal rank of first relevant result |
| Answer accuracy | Yes/no/maybe classification accuracy vs. gold label |
| Answer F1 | Free-text explanation quality vs. gold long-answer |

Full ablation results are in `src/evaluation/ablation.py` and summarized in the final report.

## Roadmap

- [ ] Data pipeline + indexing
- [ ] Baseline retrieval (BM25, dense, hybrid) + recall@k evaluation
- [ ] BioBERT reranker training
- [ ] Generation pipeline with grounded prompting
- [ ] Full ablation study
- [ ] Streamlit demo
- [ ] Final report

## Notes and limitations

This is a research/retrieval assistant over published literature, not a diagnostic or clinical decision-making tool. Answers should be treated as a summary of retrieved evidence, not medical advice.

## License

MIT