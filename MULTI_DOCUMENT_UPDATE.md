# Multi-Document Upload Update

## Overview

Both `app.py` and `RAG_Attempt.ipynb` have been updated to support **multiple document uploads** through the Gradio UI. The application no longer requires hardcoded local document paths.

## Key Changes

### 1. **app.py** - Python Script Version

#### Document Upload Tab
- New `gr.File()` component for uploading multiple PDFs, DOCX, and TXT files
- Users can select multiple files at once

#### Processing Flow
- **Upload Tab**: Users upload one or more documents
- **Process Button**: Processes all uploaded documents at once
- **Output**: Shows status of processing (chunks created, embeddings generated)
- **Query Tab**: Users can then ask questions about uploaded documents

#### New Functions
- `process_uploaded_documents(*files)`: Handles multiple file uploads, creates chunks, generates embeddings
- `load_text_file(file_obj)`: Handles file objects from Gradio (not just file paths)
- Updates to `initialize_rag_demo()`: No longer looks for hardcoded local documents

### 2. **RAG_Attempt.ipynb** - Notebook Version

#### Gradio UI Tabs
```
📤 Upload Documents Tab:
  - File upload component (multiple files)
  - Process button
  - Status output showing:
    * Number of documents uploaded
    * Number of chunks created
    * Embeddings generation progress
    * Save location confirmation

🔍 Ask Questions Tab:
  - Query input box
  - Number of sources slider
  - Model selection (optional)
  - Answer output with citations
  - Example questions
```

#### New Functions
- `process_uploaded_documents(*files)`: Identical to app.py version
- `try_load_existing_embeddings()`: Tries to load precomputed embeddings from disk
- `initialize_rag_demo()`: Simplified - no longer processes documents automatically

#### Backward Compatibility
- Can still load previously saved embeddings from `./rag_data/`
- Existing embeddings aren't deleted - documents can be added incrementally

## Usage Flow

### First Time Using the App

1. **Start the application**
   ```bash
   # For script version:
   python app.py
   
   # For notebook version:
   # Run all cells, Gradio UI launches automatically
   ```

2. **Upload Documents**
   - Open the "📤 Upload Documents" tab
   - Click "Upload Documents" and select your files
   - Can upload multiple PDFs/DOCX/TXT files at once
   - Click "⚙️ Process Documents" button

3. **Wait for Processing**
   - System will:
     1. Load each document
     2. Split into semantic chunks
     3. Generate embeddings (takes ~5 min for large documents)
     4. Save embeddings to `./rag_data/` for future use

4. **Ask Questions**
   - Once processing completes, switch to "🔍 Ask Questions" tab
   - Type your question
   - Adjust "Number of Sources" slider (1-10)
   - Click "🔍 Search"

### Subsequent Runs

- Embeddings are cached, so the app starts instantly
- Can upload new documents and click "Process" to add them to the vector store
- All documents are kept in the same vector store

## File Formats Supported

| Format | Extension | Support |
|--------|-----------|---------|
| PDF | `.pdf` | ✅ Full support with page numbers |
| Word | `.docx` | ✅ Full support |
| Text | `.txt` | ✅ Full support |

## Configuration

### Environment Variables

```bash
# No changes needed! These were already in place:
EMBEDDING_METHOD=huggingface  # or "voyage", "openai", "fastembed"
HF_TOKEN=...                   # For HuggingFace embeddings
OLLAMA_BASE_URL=...            # For local LLM
OLLAMA_MODEL=...               # Model to use
CHUNK_SIZE=1000                # Characters per chunk
CHUNK_OVERLAP=200              # Overlap between chunks
```

### Local Configuration

Still uses these settings from `DOCUMENT_CONFIG`:
- `save_dir`: Where to save embeddings (`./rag_data`)
- `chunk_size`: Characters per chunk (1000)
- `chunk_overlap`: Overlap between chunks (200)

## No More Hardcoded Paths!

### Before
```python
sample_path = "~/Documents/Books/Atomic_Habits_James_Clear.pdf"  # Hardcoded
```

### After
```python
# Users upload files via Gradio UI
# No hardcoded paths needed
# Works with any documents
```

## Testing the New System

### Quick Test
1. Run the app
2. Upload a small test document (~1-5 pages)
3. Click "Process Documents"
4. Ask a simple question like "What is this document about?"

### Full Test with Multiple Documents
1. Upload 2-3 documents
2. Process them together
3. Ask questions about specific documents
4. Verify that answers cite correct sources

## Error Handling

### Common Issues

| Error | Solution |
|-------|----------|
| "No files uploaded" | Upload at least one document |
| "Processing takes too long" | Normal for first run; embeddings are cached |
| "Port already in use" | App will auto-select next available port |
| "Out of memory" | Use fewer/smaller documents |

## Docker Deployment

No changes needed to Docker files! The app still:
- Runs on port 7860 (Gradio)
- Uses environment variables
- Mounts volumes for data persistence:
  - `./rag_data`: Vector store embeddings
  - `./.embedding_cache`: Embedding cache

## Breaking Changes

⚠️ **None!** The update is fully backward compatible:
- Existing saved embeddings still work
- App can load and use previously processed documents
- Can mix old (hardcoded) and new (uploaded) workflows

## File Changes Summary

| File | Changes |
|------|---------|
| `app.py` | Complete multi-document UI, removed hardcoded paths |
| `RAG_Attempt.ipynb` | Tab-based UI, multi-document processing, initialization simplified |
| `.dockerignore` | No changes |
| `Dockerfile` | No changes |
| `docker-compose.yml` | No changes |

## Next Steps

1. **Test locally**: Run `python app.py` and upload a document
2. **Build Docker image**: `docker-compose build`
3. **Deploy**: `docker-compose up`
4. **Use GitHub Actions**: CI/CD pipeline automatically tests and deploys

---

**Questions?** Check the Gradio UI help text or refer to the function docstrings in the code.
