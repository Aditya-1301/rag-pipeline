# RAG Pipeline

This repository implements a terminal-first Retrieval‑Augmented Generation (RAG) pipeline for document search and question answering. All code and optimizations are contained in the main notebook: `pipeline.ipynb`.

## Quickstart

Clone the repo, create a virtualenv, and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Open `pipeline.ipynb` in Jupyter or VS Code and run the cells sequentially.

## Features

- Document ingestion (PDF, TXT, DOCX)
- Semantic chunking (preserves sentence boundaries)
- Embedding generation (HuggingFace, OpenAI, Voyage, FastEmbed)
- Smart embedding reuse to avoid redundant API calls
- Vector search (FAISS)
- LLM answer generation (Ollama, OpenAI, etc.)
- Citation and source formatting in answers

## File structure

- `pipeline.ipynb` — Main notebook with the full pipeline and optimizations
- `requirements.txt` — Python dependencies
- `rag_data/` — Saved vector store, embeddings and metadata

## Environment setup

Copy and edit `.env.example` as needed. Common vars:

- `EMBEDDING_METHOD` — `huggingface`, `voyage`, `openai`, or `fastembed`
- `HF_TOKEN`, `OPENAI_API_KEY`, `VOYAGE_API_KEY` — provider keys if required
- `OLLAMA_BASE_URL` / model config for local LLM usage
- `SAVE_DIR` — defaults to `./rag_data`

## Status

Active development occurs in `pipeline.ipynb`. For optimizations, bugfixes, or new features, edit the notebook only.

Notes:
- The notebook implements a cache-aware embedding strategy (check for existing embeddings before regenerating).
- Preserve `source` and `page` metadata through the pipeline for accurate citations.
- Use `tqdm` progress bars for long operations.
- Follow environment-first configuration: CLI args > env vars > notebook defaults.
- Do not create additional files unless explicitly requested.