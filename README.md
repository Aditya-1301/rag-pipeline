# RAG Pipeline - Jupyter Notebook (FAISS + Multi-Backend Embeddings)

A **production-ready Retrieval-Augmented Generation** system built as an interactive Jupyter notebook (`RAG_Attempt.ipynb`) with:

### Key Features:

- **Fast Embeddings** (50x faster than local):
  - 🚀 **HuggingFace Inference API** (768-dim, FREE with rate limits) — Recommended
  - 🚀 **Voyage AI** (1024-dim, $0.12/1M tokens)
  - 🚀 **OpenAI** (1536-dim, $0.13/1M tokens)
  - ⚙️ **FastEmbed** (384-dim, free, local, CPU-only) — Fallback
- **Semantic Chunking**: Respects sentence boundaries, preserves page numbers
- **PyMuPDF Integration**: 3-5x faster PDF parsing with accurate page tracking
- **Smart Embedding Caching**: Detects precomputed embeddings, skips redundant API calls
- **Parallel Processing**: Multi-threaded API calls for 5-10x speedup
- **Vector Search**: Cosine similarity (FAISS IndexFlatIP) with L2 normalization
- **Local LLM Answering**: Ollama integration (gemma3, smollm2, gpt-oss)
- **Interactive UI**: Gradio demo with auto port detection
- **Source Citations**: Formatted answers with [1], [2] citations + source details
- **Persistence**: Save/load vector store and metadata (no reprocessing!)

### Workflows:

1. **First run**: Process document → Generate embeddings → Save to disk (~5 minutes)
2. **Subsequent runs**: Load cached embeddings → Query (~15 seconds, zero API calls!)
3. **Interactive mode**: Gradio web UI for testing different queries

## Setup

> Works on CPU-only boxes. No CUDA needed. All dependencies in `requirements.txt`.

```bash
# 1) Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2) Ensure pip is installed (fix for missing pip)
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt

# 4) Configure environment (optional, for API embeddings)
cp .env.example .env
# Edit .env and add your API keys:
#   HF_TOKEN (for HuggingFace — FREE, recommended)
#   VOYAGE_API_KEY (optional, fastest embeddings)
#   OPENAI_API_KEY (optional, for OpenAI embeddings or LLM)
#   OLLAMA_MODEL (local Ollama model name, default: smollm2:360m)
#   OLLAMA_BASE_URL (default: http://127.0.0.1:11434)
```

### Environment Variables (`.env`):

```bash
# Embedding Backend (choose one)
EMBEDDING_METHOD=huggingface        # Options: "huggingface" (FREE), "voyage", "openai", "fastembed"
HF_TOKEN=hf_your_token_here         # Get free token: https://huggingface.co/settings/tokens

# Optional: Cloud embeddings
VOYAGE_API_KEY=pa_your_key          # Voyage AI (fastest, paid)
OPENAI_API_KEY=sk_your_key          # OpenAI (fast, paid)

# Local LLM via Ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=smollm2:360m           # Or: gemma3:270m, gpt-oss:20b (cloud)
OLLAMA_API_KEY=optional_for_cloud   # Required only for cloud models

# Document Processing
CHUNK_SIZE=1000                     # Characters per chunk
CHUNK_OVERLAP=200                   # Character overlap between chunks
TOP_K=5                             # Number of sources to retrieve
```

## Quickstart

### Option 1: Interactive Jupyter Notebook (Recommended)

```bash
# 1) Open the notebook
jupyter notebook RAG_Attempt.ipynb

# 2) Run cells in order:
#    - Cell 1-2: Load environment & configure embeddings
#    - Cell 3-7: Import libraries & initialize embedding backends
#    - Cell 8-10: Load document (PDF/txt/docx), chunk it, and generate embeddings
#    - Cell 11-12: Build FAISS vector store and metadata
#    - Cell 13-14: Retrieve documents and generate answers
#    - Cell 15-16: Launch Gradio web UI (or query programmatically)

# 3) Query the document via Gradio UI or Python:
# query = "What are the Four Laws of Behavior Change?"
# result = execute_query(query, vector_store, metadata_store)
# display_result(result)
```

### Option 2: Gradio Web Demo

The notebook includes a **Gradio interface** that automatically launches:

```python
# After running the pipeline cells:
# - Opens interactive web UI at http://127.0.0.1:7860
# - Adjust "Number of Sources" (1-10) to control answer length
# - Try example questions with one click
# - Automatically detects available ports (handles conflicts)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 DOCUMENT INGESTION                          │
│  load_document() → PDF/txt/docx parsed with metadata        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              SEMANTIC CHUNKING (Smart!)                     │
│  chunk_document_semantic() respects sentence boundaries     │
│  Config: CHUNK_SIZE=1000 chars, OVERLAP=200 chars           │
│  Preserves page numbers for accurate citations              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│           EMBEDDING GENERATION (OPTIMIZED)                  │
│  embed_texts_optimized() with:                              │
│  • EmbeddingCache (file-based MD5 hashing, 10x+ speedup)    │
│  • ParallelHFEmbedder (4-worker thread pool)                │
│  • LocalFastEmbedder (batch processing)                     │
│  • Smart skip: detects & loads precomputed embeddings       │
│  APIs: HuggingFace, Voyage, OpenAI, or local FastEmbed      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│            VECTOR STORAGE & METADATA                        │
│  FAISS IndexFlatIP (cosine similarity, L2 norm)             │
│  Storage: ./rag_data/{faiss_index.bin, metadata.pkl}        │
│  Fast loading: 1-2 seconds from disk                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVAL & ANSWER GENERATION                  │
│  retrieve_documents(query, k=TOP_K)                         │
│  generate_answer() with local Ollama                        │
│  format_answer_with_sources() → citations [1], [2]          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Example:

```
User Query: "What are the Four Laws of Behavior Change?"
    ↓
embed_texts([query]) → 768-dimensional vector via HuggingFace
    ↓
vector_store.search(query_vector, k=5) → top 5 similar chunks
    ↓
retrieved_chunks = [
    {"text": "...cue, craving, response, reward...", "page": 42, "source": "Atomic_Habits.pdf"},
    {"text": "...the habit loop binds them together...", "page": 45, "source": "Atomic_Habits.pdf"},
    ...
]
    ↓
generate_answer(query, retrieved_chunks) via local Ollama
    ↓
"The Four Laws are cue, craving, response, and reward [1].
  Each law represents a stage in the habit loop [2].

📚 SOURCES:
[1] Atomic_Habits.pdf (Page 42): "...cue, craving, response..."
[2] Atomic_Habits.pdf (Page 45): "...habit loop..."
```

## File Structure

```
RAG_Attempt.ipynb          # Main interactive notebook
├─ Cell 1-7: Setup & environment loading
├─ Cell 8-10: Document loading (PDF/txt/docx) & chunking
├─ Cell 11-12: Embedding generation with optimizations
├─ Cell 13-14: Vector store & retrieval
├─ Cell 15-16: Answer generation & formatting
└─ Cell 17-19: Gradio UI demo & testing

pipeline.ipynb             # Experimental/development notebook

.env.example               # Configuration template
requirements.txt           # All dependencies
README.md                  # This file

rag_data/                  # Generated data (created after first run)
  ├─ faiss_index.bin       # Vector index (~2.6 MB per 660 chunks)
  ├─ chunk_ids.pkl         # Chunk ID mapping
  ├─ metadata.pkl          # Document metadata & source info
  └─ .embedding_cache/     # File-based embedding cache
```

## Next Steps 

1. **Query Caching**: Cache popular Q&A pairs
2. **Re-ranking**: Add cross-encoders for better retrieval
3. **Evaluation**: Implement BLEU/ROUGE metrics
4. **Multi-document**: Support directory ingestion
5. **FastAPI Wrapper**: Deploy as REST API
6. **Monitoring**: Add Langfuse/LLM observability

## Support

For issues or questions:

1. Check `.env` file is correctly configured
2. Verify API keys aren't expired
3. Review cell outputs in notebook for error messages
4. See Troubleshooting section above
