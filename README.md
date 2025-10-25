# RAG Terminal & Gradio Demo (FAISS + FastEmbed, Ollama, HuggingFace, OpenAI)

A Retrieval-Augmented Generation pipeline supporting:

- **Embeddings:** FastEmbed (ONNX, CPU), HuggingFace API, OpenAI, Voyage AI
- **Index:** FAISS (cosine similarity)
- **Chunking:** semantic (sentence-aware) with overlap
- **Answering:** Local Ollama LLM (smollm2, gemma3, etc.), OpenAI, HuggingFace
- **Web Demo:** Gradio UI (interactive Q&A with citations)
- **Persistence:** Save/load vector store and metadata for fast reuse
- **Progress Tracking:** tqdm progress bars for long operations

Supports PDF, TXT, MD, DOCX documents. Embedding/LLM backend is configurable via `.env`.

## Install

> Works on CPU-only boxes. No CUDA needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
cp .env.example .env   # then edit if needed
```

If you prefer OpenAI embeddings or LLM answering, fill `OPENAI_API_KEY` in `.env` and set `EMBEDDING_BACKEND=openai`.

Works on CPU-only boxes. No CUDA needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
cp .env.example .env   # then edit if needed
```

- For Gradio demo: `gradio` is now required (see requirements.txt)
- For HuggingFace/Voyage/OpenAI: fill in API keys in `.env`

## Quickstart

### CLI Workflow

```bash
# 0) Initialize storage (creates .rag/ folder)
python rag.py init

# 1) Ingest some text/files
python rag.py ingest --text "Transformers replace recurrence with self-attention."
python rag.py ingest --file docs/notes.txt
python rag.py ingest --glob "docs/**/*.txt"

# 2) Query
python rag.py query "What is self-attention and why is it useful?"

# 2b) Query with LLM drafting (if OPENAI_API_KEY set)
python rag.py query "What is self-attention and why is it useful?" --llm openai --max-sources 4
```

### Gradio Demo (Recommended)

```bash
# 1) Run all notebook cells in RAG_Attempt.ipynb
# 2) The Gradio demo cell will launch a local and public web UI
#    - Local: http://127.0.0.1:7860 (or next available port)
#    - Public: (auto-generated share link)
# 3) Ask questions, adjust number of sources, and view citations interactively
```

## Files & Folders

```
rag.py                # CLI (Typer)
raglib/
  config.py           # settings from env
  chunker.py          # token-aware chunking
  embed.py            # fastembed / openai
  store.py            # docstore + FAISS index
  retriever.py        # add/search
  answer.py           # draft & format
.rag/
  faiss.index         # vector index
  docstore.jsonl      # chunk texts + metadata
.env.example          # config
requirements.txt
README.md
```

## Notes

- Default backend uses **fastembed** (CPU) → small download (ONNX) on first run.
- Initial MVP only supports **.txt/.md**. Add PDF/HTML parsing later.
- This is meant to be minimal and robust; once happy, you can wrap it in Streamlit/Gradio/FastAPI.
