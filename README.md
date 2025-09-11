# RAG Terminal MVP (FAISS + FastEmbed, no PyTorch)

A **terminal-only** Retrieval-Augmented Generation starter that runs fully offline for retrieval:
- **Embeddings:** `fastembed` (ONNX, CPU) by default — no PyTorch/CUDA.
- **Index:** `faiss-cpu`
- **Chunking:** token-aware (tiktoken) with char fallback
- **Answering:** 
  - default: **extractive** (prints top passages with citations)
  - optional: **LLM draft** using OpenAI if `OPENAI_API_KEY` is set

This keeps the workflow simple: **ingest → query** from the terminal. You can later port it to Streamlit/Gradio or FastAPI+React.

## Install

> Works on CPU-only boxes. No CUDA needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
cp .env.example .env   # then edit if needed
```

If you prefer OpenAI embeddings or LLM answering, fill `OPENAI_API_KEY` in `.env` and set `EMBEDDING_BACKEND=openai`.

## Quickstart

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
