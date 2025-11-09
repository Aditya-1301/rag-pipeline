# Deploying to HuggingFace Spaces

This guide explains how to deploy the RAG Pipeline to HuggingFace Spaces.

## Important Note About Environment Variables

**The `.env` file is gitignored and will NOT be pushed to HuggingFace Spaces.**

Instead, you need to set environment variables as **Repository Secrets** in your HF Space.

## Step-by-Step Deployment

### 1. Create a HuggingFace Space

1. Go to https://huggingface.co/new-space
2. Choose:
   - **Name:** Your space name (e.g., `SimpleRAGPipeline`)
   - **License:** MIT or your choice
   - **SDK:** Gradio
   - **Hardware:** CPU basic (free tier)

### 2. Push Your Code

```bash
# Add HF Space as remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push to HF Space
git push space main
```

### 3. Configure Repository Secrets

Go to your Space's **Settings → Repository secrets** and add these secrets:

#### Required Secrets:

| Secret Name   | Value         | Description                                                                    |
| ------------- | ------------- | ------------------------------------------------------------------------------ |
| `HF_TOKEN`    | `hf_...`      | Your HuggingFace token ([Get it here](https://huggingface.co/settings/tokens)) |
| `LLM_BACKEND` | `huggingface` | Use HF Inference API for LLM generation                                        |

#### Optional Secrets (if using alternatives):

| Secret Name      | Value    | Description                   |
| ---------------- | -------- | ----------------------------- |
| `OPENAI_API_KEY` | `sk-...` | If using OpenAI instead of HF |
| `LLM_BACKEND`    | `openai` | If using OpenAI               |

### 4. Verify Configuration

After pushing, check the Space logs to ensure:

```
✅ Backend auto-detection: using 'huggingface'
✅ HF_TOKEN configured
✅ Model: HuggingFaceTB/SmolLM2-1.7B-Instruct
```

If you see:

```
⚠️ WARNING: HF_TOKEN not set!
```

Then you need to add `HF_TOKEN` as a repository secret.

## Default Configuration for HF Spaces

The application is pre-configured to work on HF Spaces with these defaults:

- **LLM Backend:** HuggingFace Inference API (when `HF_TOKEN` is set)
- **Model:** SmolLM2-1.7B-Instruct (fast on CPU)
- **Embeddings:** FastEmbed (local, no API calls)
- **Web Search:** Disabled (duckduckgo-search causes issues)

## Troubleshooting

### "Error generating answer: Failed to connect to Ollama"

**Cause:** `LLM_BACKEND` is not set, so it defaults to `auto` which tries Ollama (not available on HF Spaces).

**Fix:** Add `LLM_BACKEND=huggingface` as a repository secret.

### "HF_TOKEN not configured"

**Cause:** The `HF_TOKEN` secret is not set.

**Fix:** Add your HuggingFace token as a repository secret named `HF_TOKEN`.

### "Rate limit exceeded"

**Cause:** Too many requests to HF Inference API (free tier has limits).

**Fix:**

- Upgrade to HF Pro for higher rate limits
- Or switch to OpenAI: set `LLM_BACKEND=openai` and `OPENAI_API_KEY` secrets

## Local Development vs HF Spaces

### Local Development (.env file):

```bash
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=smollm2:360m
```

### HF Spaces (Repository Secrets):

```
LLM_BACKEND=huggingface
HF_TOKEN=hf_...
```

## Architecture on HF Spaces

```
User Query
    ↓
FastEmbed (local) → Generate embeddings
    ↓
FAISS → Retrieve relevant chunks
    ↓
HuggingFace Inference API → Generate answer
    ↓
Response with citations
```

**Benefits:**

- ✅ Runs entirely on CPU (no GPU needed)
- ✅ Free tier available (with rate limits)
- ✅ No local installation required
- ✅ Automatic scaling and hosting

## Cost Analysis

| Component  | HF Spaces Free    | Cost                |
| ---------- | ----------------- | ------------------- |
| Hosting    | CPU basic         | Free                |
| Embeddings | FastEmbed (local) | Free                |
| LLM        | HF Inference API  | Free (rate-limited) |
| **Total**  |                   | **$0/month**        |

For higher throughput, upgrade to HF Pro or use Zero GPU spaces.

## Resources

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
- [SmolLM2 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
