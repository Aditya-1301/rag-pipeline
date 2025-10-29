#!/usr/bin/env python3
"""
RAG Terminal Application
Retrieval-Augmented Generation system with Gradio UI
"""

import os
import json
import numpy as np
import faiss
from fastembed import TextEmbedding
from ollama import Client
from tqdm.auto import tqdm
import pickle
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fitz  # PyMuPDF
import docx
from typing import List, Tuple
import re
import uuid
import gradio as gr
import socket
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=False)

print("=" * 70)
print("RAG TERMINAL APPLICATION")
print("=" * 70)
print("\nEnvironment variables loaded from .env:")
print(f"  ✓ OLLAMA_API_KEY present: {bool(os.environ.get('OLLAMA_API_KEY'))}")
print(f"  ✓ HF_TOKEN present: {bool(os.environ.get('HF_TOKEN'))}")
print(f"  ✓ OPENAI_API_KEY present: {bool(os.environ.get('OPENAI_API_KEY'))}")
print(f"  ✓ VOYAGE_API_KEY present: {bool(os.environ.get('VOYAGE_API_KEY'))}")
print()

# ===== API CLIENT INITIALIZATION =====

# API-based embeddings (faster alternatives)
try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configuration: Choose embedding method
EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "huggingface")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Local LLMs via Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm2:360m")
OLLAMA_MODEL_CLOUD = os.getenv("OLLAMA_MODEL_CLOUD", "gpt-oss:20b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", None)
client = Client(host=OLLAMA_BASE_URL)

print(f"Using Ollama model: {OLLAMA_MODEL_CLOUD}")
print(f"OLLAMA_API_KEY configured: {bool(OLLAMA_API_KEY)}")

# Initialize embedding client based on chosen method
if EMBEDDING_METHOD == "voyage" and VOYAGE_AVAILABLE and VOYAGE_API_KEY:
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    EMBEDDING_DIM = 1024
    print("Using Voyage AI embeddings (1024-dim)")
elif EMBEDDING_METHOD == "openai" and OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    EMBEDDING_DIM = 1536
    print("Using OpenAI embeddings (1536-dim)")
elif EMBEDDING_METHOD == "huggingface" and HF_AVAILABLE and HF_TOKEN:
    hf_client = InferenceClient(token=HF_TOKEN)
    HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIM = 768
    print(f"Using HuggingFace Inference API with {HF_MODEL} (768-dim)")
else:
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    EMBEDDING_DIM = 384
    EMBEDDING_METHOD = "fastembed"
    print("Using local FastEmbed (384-dim) - Set API keys for faster embeddings")

print("Ollama client ready; embedding system initialized.\n")

# ===== EMBEDDING OPTIMIZATION LAYER =====

class EmbeddingCache:
    """File-based cache for embeddings using content hash as key."""
    
    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
    
    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> np.ndarray:
        key = self._hash_text(text)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file)
            self.memory_cache[key] = embedding
            return embedding
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        key = self._hash_text(text)
        self.memory_cache[key] = embedding
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)


class ParallelHFEmbedder:
    """HuggingFace embeddings with parallel processing (4 workers)."""
    
    def __init__(self, model: str, api_key: str, num_workers: int = 4):
        self.model = model
        self.api_key = api_key
        self.num_workers = num_workers
        self.client = InferenceClient(token=api_key)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        results = [None] * len(texts)
        lock = threading.Lock()
        
        def embed_single(idx: int, text: str):
            try:
                embedding = self.client.feature_extraction(text, model=self.model)
                embedding = np.array(embedding, dtype="float32")
                if embedding.ndim > 1:
                    embedding = embedding[0]
                with lock:
                    results[idx] = embedding
            except Exception as e:
                print(f"Error embedding text {idx}: {e}")
                with lock:
                    results[idx] = np.zeros(768, dtype="float32")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(embed_single, idx, text) 
                      for idx, text in enumerate(texts)]
            for future in futures:
                future.result()
        
        return np.array([r for r in results if r is not None], dtype="float32")


class LocalFastEmbedder:
    """Local sentence-transformers embeddings with batch processing."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            print("⚠️  sentence-transformers not available")
            self.available = False
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not self.available:
            return None
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings, dtype="float32")


def embed_texts_optimized(texts: list[str], batch_size: int = 100, 
                         use_cache: bool = True, 
                         use_parallel: bool = True,
                         use_local_fast: bool = False) -> np.ndarray:
    """Optimized embedding generation with caching and parallel processing."""
    
    cache = EmbeddingCache() if use_cache else None
    uncached_texts = []
    uncached_indices = []
    cached_embeddings = {}
    
    if use_cache:
        for i, text in enumerate(texts):
            cached_emb = cache.get(text)
            if cached_emb is not None:
                cached_embeddings[i] = cached_emb
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if cached_embeddings:
            print(f"✓ Found {len(cached_embeddings)} cached embeddings")
    else:
        uncached_texts = texts
        uncached_indices = list(range(len(texts)))
    
    if not uncached_texts:
        print("✓ All embeddings loaded from cache!")
        return np.array([cached_embeddings[i] for i in range(len(texts))], dtype="float32")
    
    # Try local fast embedding first if requested
    if use_local_fast:
        print(f"Embedding {len(uncached_texts)} texts using local sentence-transformers...")
        local_embedder = LocalFastEmbedder()
        if local_embedder.available:
            new_embeddings = local_embedder.embed_batch(uncached_texts)
            if new_embeddings is not None:
                if use_cache:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        cache.put(text, embedding)
                
                all_embeddings = np.zeros((len(texts), new_embeddings.shape[1]), dtype="float32")
                for i, emb in cached_embeddings.items():
                    all_embeddings[i] = emb
                for idx, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[idx] = emb
                return all_embeddings
    
    # Embed uncached texts
    if use_parallel and EMBEDDING_METHOD == "huggingface" and HF_AVAILABLE and HF_TOKEN:
        print(f"Embedding {len(uncached_texts)} texts using parallel HuggingFace (4 workers)...")
        parallel_embedder = ParallelHFEmbedder(HF_MODEL, HF_TOKEN, num_workers=4)
        new_embeddings = parallel_embedder.embed_batch(uncached_texts)
    else:
        print(f"Embedding {len(uncached_texts)} texts...")
        new_embeddings = []
        
        for i in tqdm(range(0, len(uncached_texts), batch_size), desc="Embedding"):
            batch = uncached_texts[i:i + batch_size]
            
            if EMBEDDING_METHOD == "voyage":
                result = voyage_client.embed(batch, model="voyage-2", input_type="document")
                batch_embeddings = np.array(result.embeddings, dtype="float32")
            elif EMBEDDING_METHOD == "openai":
                result = openai_client.embeddings.create(input=batch, model="text-embedding-3-small")
                batch_embeddings = np.array([e.embedding for e in result.data], dtype="float32")
            elif EMBEDDING_METHOD == "huggingface":
                batch_embeddings = []
                for text in batch:
                    result = hf_client.feature_extraction(text, model=HF_MODEL)
                    embedding = np.array(result, dtype="float32")
                    if embedding.ndim > 1:
                        embedding = embedding[0]
                    batch_embeddings.append(embedding)
                batch_embeddings = np.array(batch_embeddings, dtype="float32")
            else:  # fastembed
                vecs = list(embedder.embed(batch))
                batch_embeddings = np.array([np.asarray(v, dtype="float32") for v in vecs])
            
            new_embeddings.append(batch_embeddings)
        
        new_embeddings = np.vstack(new_embeddings)
    
    # Cache the new embeddings
    if use_cache:
        for text, embedding in zip(uncached_texts, new_embeddings):
            cache.put(text, embedding)
    
    # Merge cached and new embeddings
    all_embeddings = np.zeros((len(texts), new_embeddings.shape[1]), dtype="float32")
    for i, emb in cached_embeddings.items():
        all_embeddings[i] = emb
    for idx, emb in zip(uncached_indices, new_embeddings):
        all_embeddings[idx] = emb
    
    return all_embeddings


print("✓ Embedding optimization layer loaded\n")

# ===== DOCUMENT LOADING =====

def load_document(file_path: str):
    """Loads content from various document types."""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        return load_pdf(file_path)
    elif file_extension in ['.md', '.txt']:
        return load_text(file_path)
    elif file_extension == '.docx':
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def load_pdf(file_path: str) -> List[Tuple[str, int]]:
    """Loads text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text = ' '.join(text.split())
            
            if text.strip():
                pages_data.append((text, page_num + 1))
        
        doc.close()
        print(f"Loaded {len(pages_data)} pages from PDF")
        return pages_data
    except Exception as e:
        print(f"Error loading PDF file {file_path}: {e}")
        return []


def load_text(file_path: str) -> str:
    """Loads text from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading text file {file_path}: {e}")
        return None


def load_docx(file_path: str) -> str:
    """Loads text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error loading DOCX file {file_path}: {e}")
        return None


# ===== CHUNKING =====

def chunk_document_semantic(document_text: str, chunk_size: int = 1000, 
                           overlap: int = 200, page_number: int = None) -> List[Tuple[str, int]]:
    """Splits a document into semantic chunks that respect sentence boundaries."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(document_text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append((current_chunk.strip(), page_number))
            
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                first_sentence_start = overlap_text.find('. ')
                if first_sentence_start != -1:
                    overlap_text = overlap_text[first_sentence_start + 2:]
                current_chunk = overlap_text + sentence + " "
            else:
                current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append((current_chunk.strip(), page_number))
    
    return chunks


def chunk_pdf_pages(pages_data: List[Tuple[str, int]], chunk_size: int = 1000, 
                    overlap: int = 200):
    """Chunks PDF pages while preserving page number information."""
    all_chunks = []
    for page_text, page_num in tqdm(pages_data, desc="Chunking pages"):
        page_chunks = chunk_document_semantic(page_text, chunk_size, overlap, page_num)
        all_chunks.extend(page_chunks)
    return all_chunks


# ===== METADATA STORAGE =====

class DocumentChunk:
    def __init__(self, text: str, source: str, page_number: int = None, chunk_id: str = None):
        self.chunk_id = chunk_id if chunk_id is not None else str(uuid.uuid4())
        self.text = text
        self.source = source
        self.page_number = page_number
        self.embedding = None


class MetadataStore:
    def __init__(self):
        self.chunks = {}

    def add_chunk(self, chunk: DocumentChunk):
        self.chunks[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> DocumentChunk:
        return self.chunks.get(chunk_id)

    def get_all_chunks(self) -> list[DocumentChunk]:
        return list(self.chunks.values())

    def index_chunks(self):
        print(f"Indexed {len(self.chunks)} chunks in metadata store.")

    def save(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Metadata store saved to {save_path}")

    @classmethod
    def load(cls, save_path: str, verbose: bool = True):
        instance = cls()
        with open(save_path, 'rb') as f:
            instance.chunks = pickle.load(f)
        if verbose:
            print(f"Metadata store loaded from {save_path} ({len(instance.chunks)} chunks)")
        return instance


# ===== EMBEDDING GENERATION =====

def generate_embeddings(chunks: list[DocumentChunk], batch_size: int = 100, 
                       use_optimizations: bool = True):
    """Generates embeddings for chunks using configured method with optimizations."""
    texts = [chunk.text for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks using optimized {EMBEDDING_METHOD}...")
    embeddings = embed_texts_optimized(
        texts, 
        batch_size=batch_size,
        use_cache=use_optimizations,
        use_parallel=use_optimizations,
        use_local_fast=False
    )
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    
    print(f"✓ Generated {len(embeddings)} embeddings ({EMBEDDING_DIM}-dimensional)")


# ===== VECTOR STORE =====

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_ids = []

    def add_vectors(self, embeddings: np.ndarray, chunk_ids: list[str]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        print(f"Added {len(chunk_ids)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.chunk_ids):
                results.append(self.chunk_ids[i])
        return results

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        index_path = os.path.join(save_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        metadata_path = os.path.join(save_dir, "chunk_ids.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({'chunk_ids': self.chunk_ids, 'dimension': self.dimension}, f)
        print(f"Vector store saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, verbose: bool = True):
        index_path = os.path.join(save_dir, "faiss_index.bin")
        metadata_path = os.path.join(save_dir, "chunk_ids.pkl")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        instance = cls(metadata['dimension'])
        instance.index = faiss.read_index(index_path)
        instance.chunk_ids = metadata['chunk_ids']
        
        if verbose:
            print(f"Vector store loaded from {save_dir} ({instance.index.ntotal} vectors)")
        return instance


# ===== RETRIEVAL =====

def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Simple embedding function for queries."""
    if not isinstance(texts, list):
        texts = [texts]
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        if EMBEDDING_METHOD == "voyage":
            result = voyage_client.embed(batch, model="voyage-2", input_type="query")
            batch_embeddings = np.array(result.embeddings, dtype="float32")
        elif EMBEDDING_METHOD == "openai":
            result = openai_client.embeddings.create(input=batch, model="text-embedding-3-small")
            batch_embeddings = np.array([e.embedding for e in result.data], dtype="float32")
        elif EMBEDDING_METHOD == "huggingface":
            batch_embeddings = []
            for text in batch:
                result = hf_client.feature_extraction(text, model=HF_MODEL)
                embedding = np.array(result, dtype="float32")
                if embedding.ndim > 1:
                    embedding = embedding[0]
                batch_embeddings.append(embedding)
            batch_embeddings = np.array(batch_embeddings, dtype="float32")
        else:  # fastembed
            vecs = list(embedder.embed(batch))
            batch_embeddings = np.array([np.asarray(v, dtype="float32") for v in vecs])
        
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings) if len(embeddings) > 1 else embeddings[0]


def retrieve_documents(query: str, vector_store: VectorStore, 
                      metadata_store: MetadataStore, k: int = 5):
    """Retrieves the top K most relevant document chunks for a given query."""
    qv = embed_texts([query])
    retrieved_chunk_ids = vector_store.search(qv[0], k=k)
    retrieved_chunks = [metadata_store.get_chunk(chunk_id) 
                       for chunk_id in retrieved_chunk_ids 
                       if metadata_store.get_chunk(chunk_id) is not None]
    return retrieved_chunks


# ===== ANSWER GENERATION =====

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer ONLY using the provided sources.\n"
    "Cite evidence with bracketed indices like [1], [2]. If unsure, say you don't know.\n"
    "Keep it concise: 3-6 sentences."
)

TOKEN_LIMIT_BASE = 512
TOKEN_LIMIT_PER_SOURCE = 200
TOKEN_LIMIT_MAX = 2048


def generate_answer(query: str, retrieved_chunks: list, model_name: str = None, 
                   stream: bool = False):
    """Generate an answer from Ollama or Ollama cloud."""
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL", "smollm2:360m")

    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        src = getattr(chunk, "source", f"doc_{i}")
        chunk_text_raw = getattr(chunk, "text", "")
        chunk_text = str(chunk_text_raw) if chunk_text_raw else ""
        context_lines.append(f"[{i}] {src}:\n{chunk_text[:800]}")

    user = f"Question: {query}\n\nSources:\n" + "\n\n".join(context_lines) + "\n\nAnswer:"

    num_sources = len(retrieved_chunks)
    num_predict = min(TOKEN_LIMIT_BASE + (num_sources * TOKEN_LIMIT_PER_SOURCE), TOKEN_LIMIT_MAX)
    
    print(f"[LLM Config] Sources: {num_sources} | Max tokens: {num_predict}")

    if model_name == OLLAMA_MODEL_CLOUD:
        if not OLLAMA_API_KEY:
            raise ValueError("OLLAMA_API_KEY is not set but cloud model was requested.")
        cloud_client = Client(host="https://ollama.com", 
                            headers={"Authorization": "Bearer " + OLLAMA_API_KEY})
        chosen_client = cloud_client
    else:
        chosen_client = client

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    try:
        if stream:
            resp_stream = chosen_client.chat(model=model_name, messages=messages, 
                                            stream=True, 
                                            options={"temperature": 0.2, "num_predict": num_predict})
            full_text = ""
            try:
                for part in resp_stream:
                    if isinstance(part, dict) and "message" in part:
                        delta = part["message"].get("content", "")
                    elif isinstance(part, str):
                        delta = part
                    else:
                        delta = ""
                    print(delta, end="", flush=True)
                    full_text += delta
                print()
                return full_text.strip()
            except TypeError:
                pass

        resp = chosen_client.chat(model=model_name, messages=messages, 
                                 options={"temperature": 0.2, "num_predict": num_predict})
        if isinstance(resp, dict):
            message = resp.get("message")
            if isinstance(message, dict):
                return message.get("content", "").strip()
            return resp.get("content", "").strip()
        if isinstance(resp, str):
            return resp.strip()
        return str(resp)

    except Exception as e:
        return f"Error generating answer: {e}"


def format_answer_with_sources(answer_text: str, retrieved_chunks: list) -> str:
    """Format answer with source citations [1], [2], etc."""
    if not retrieved_chunks or not answer_text.strip():
        return answer_text
    
    sources_section = "\n📚 SOURCES:\n"
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        source_name = getattr(chunk, "source", f"Document {i}")
        page_num = getattr(chunk, "page_number", None)
        chunk_text = str(getattr(chunk, "text", "")) or "(No text content)"
        
        snippet = chunk_text[:200].replace("\n", " ")
        if len(chunk_text) > 200:
            snippet += "..."
        
        page_info = f" (Page {page_num})" if page_num else ""
        sources_section += f"\n[{i}] {source_name}{page_info}:\n"
        sources_section += f'    "{snippet}"\n'
    
    return f"{answer_text}\n{sources_section}"


# ===== CONFIGURATION =====

DOCUMENT_CONFIG = {
    "sample_path": os.getenv("SAMPLE_DOCUMENT_PATH", 
                            "/home/agupta/Documents/Books/Atomic_Habits_James_Clear.pdf"),
    "save_dir": os.getenv("SAVE_DIR", "./rag_data"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
    "top_k": int(os.getenv("TOP_K", "5")),
}

MODEL_CONFIG = {
    "ollama_local": os.getenv("OLLAMA_MODEL", "smollm2:360m"),
    "ollama_cloud": os.getenv("OLLAMA_MODEL_CLOUD", "gpt-oss:20b-cloud"),
    "api_key": os.getenv("OLLAMA_API_KEY", None),
}

TOKEN_CONFIG = {
    "base": int(os.getenv("TOKEN_LIMIT_BASE", "512")),
    "per_source": int(os.getenv("TOKEN_LIMIT_PER_SOURCE", "200")),
    "max": int(os.getenv("TOKEN_LIMIT_MAX", "2048")),
}

DEFAULT_OLLAMA_MODEL = (MODEL_CONFIG["ollama_cloud"] 
                       if MODEL_CONFIG["api_key"] 
                       else MODEL_CONFIG["ollama_local"])


def extract_content(resp):
    """Extract answer text from various response formats."""
    if resp is None:
        return ""
    
    if isinstance(resp, dict):
        msg = resp.get("message") or resp.get("content")
        if isinstance(msg, dict):
            return msg.get("content", "").strip()
        if isinstance(msg, str):
            return msg.strip()
        return str(resp).strip()
    
    if not isinstance(resp, str):
        return str(resp).strip()
    
    s = resp
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            msg = parsed.get("message")
            if isinstance(msg, dict):
                return msg.get("content", "").strip()
    except Exception:
        pass
    
    m = re.search(r"message=Message\([^)]*content=(?P<q>['\"])(?P<content>.*?)(?P=q)", 
                 s, re.DOTALL)
    if m:
        return m.group("content").strip()
    
    m2 = re.search(r"content=(?P<q>['\"])(?P<content>.*?)(?P=q)", s, re.DOTALL)
    if m2:
        return m2.group("content").strip()
    
    return s.strip()


def execute_query(query: str, vector_store: VectorStore, metadata_store: MetadataStore, 
                 model_name: str = None, top_k: int = None, verbose: bool = True) -> dict:
    """End-to-end query execution."""
    if model_name is None:
        model_name = DEFAULT_OLLAMA_MODEL
    if top_k is None:
        top_k = DOCUMENT_CONFIG["top_k"]
    
    if verbose:
        print(f"📖 Query: {query}")
    
    retrieved_chunks = retrieve_documents(query, vector_store, metadata_store, k=top_k)
    if verbose:
        print(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")
        print(f"Generating answer with model: {model_name}")
    
    raw_response = generate_answer(query, retrieved_chunks, model_name=model_name, stream=False)
    answer_text = extract_content(raw_response)
    formatted_output = format_answer_with_sources(answer_text, retrieved_chunks)
    
    return {
        "query": query,
        "answer": answer_text,
        "formatted_answer": formatted_output,
        "retrieved_chunks": retrieved_chunks,
        "model": model_name
    }


def display_result(result: dict):
    """Pretty print query result."""
    print("\n" + "=" * 70)
    print("ANSWER WITH FORMATTED SOURCES:")
    print("=" * 70)
    print(result["formatted_answer"])
    print("=" * 70)


# ===== PIPELINE FUNCTIONS =====

def check_if_embeddings_exist(save_dir: str) -> bool:
    """Check if embeddings and metadata already exist from a previous run."""
    required_files = [
        os.path.join(save_dir, "faiss_index.bin"),
        os.path.join(save_dir, "chunk_ids.pkl"),
        os.path.join(save_dir, "metadata.pkl")
    ]
    
    all_exist = all(os.path.exists(f) for f in required_files)
    
    if all_exist:
        try:
            test_vs = VectorStore.load(save_dir, verbose=False)
            test_ms = MetadataStore.load(os.path.join(save_dir, "metadata.pkl"), verbose=False)
            if test_vs.index.ntotal > 0 and len(test_ms.get_all_chunks()) > 0:
                return True
        except Exception as e:
            print(f"⚠️  Embeddings exist but failed to load: {e}")
    
    return False


def skip_embedding_if_exists(save_dir: str, sample_document_path: str):
    """Smart pipeline state manager."""
    if check_if_embeddings_exist(save_dir):
        print("\n" + "=" * 70)
        print("⚡ FOUND PRECOMPUTED EMBEDDINGS - SKIPPING REDUNDANT API CALLS")
        print("=" * 70)
        print(f"✓ Embeddings already exist in {save_dir}")
        print(f"✓ Skipping: document loading, chunking, and embedding generation")
        print(f"✓ Directly loading from disk...\n")
        
        vector_store = VectorStore.load(save_dir, verbose=False)
        metadata_store = MetadataStore.load(os.path.join(save_dir, "metadata.pkl"), verbose=False)
        
        print(f"✓ Loaded {vector_store.index.ntotal} precomputed embeddings")
        print(f"✓ Loaded {len(metadata_store.get_all_chunks())} document chunks")
        print(f"✓ Ready for querying!\n")
        
        return True, vector_store, metadata_store
    else:
        print(f"\n📄 No precomputed embeddings found. Processing document: {os.path.basename(sample_document_path)}")
        return False, None, None


# ===== DEMO PREPARATION =====

_demo_state = {
    "vector_store": None,
    "metadata_store": None,
    "initialized": False,
    "documents_processed": [],
}


def process_uploaded_documents(files) -> str:
    """Process uploaded document files and create vector store."""
    if files is None or len(files) == 0:
        return "⚠️ No files uploaded. Please upload at least one document."
    
    try:
        print(f"\n📂 Processing {len(files)} uploaded document(s)...")
        
        # Initialize stores
        ms = MetadataStore()
        all_chunks = []
        
        # Process each uploaded file
        for file_obj in files:
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
            print(f"  Processing: {os.path.basename(file_path)}")
            
            try:
                document_data = load_document(file_path)
                is_pdf = isinstance(document_data, list)
                
                if is_pdf:
                    chunks_data = chunk_pdf_pages(
                        document_data,
                        chunk_size=DOCUMENT_CONFIG["chunk_size"],
                        overlap=DOCUMENT_CONFIG["chunk_overlap"]
                    )
                else:
                    chunks_data = chunk_document_semantic(
                        document_data,
                        chunk_size=DOCUMENT_CONFIG["chunk_size"],
                        overlap=DOCUMENT_CONFIG["chunk_overlap"]
                    )
                
                # Create chunks with metadata
                for chunk_text, page_num in chunks_data:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        source=os.path.basename(file_path),
                        page_number=page_num
                    )
                    ms.add_chunk(chunk)
                    all_chunks.append(chunk)
                
                print(f"    ✓ Extracted {len(chunks_data)} chunks")
                
            except Exception as e:
                print(f"    ⚠️ Error processing {os.path.basename(file_path)}: {e}")
                continue
        
        if not all_chunks:
            return "❌ No content could be extracted from uploaded files."
        
        # Generate embeddings for all chunks
        print(f"\n🔄 Generating embeddings for {len(all_chunks)} total chunks...")
        generate_embeddings(all_chunks, batch_size=100, use_optimizations=True)
        
        # Create vector store
        vs = VectorStore(EMBEDDING_DIM)
        embeddings = np.array([chunk.embedding for chunk in all_chunks])
        chunk_ids = [chunk.chunk_id for chunk in all_chunks]
        vs.add_vectors(embeddings, chunk_ids)
        
        # Update global state
        _demo_state["vector_store"] = vs
        _demo_state["metadata_store"] = ms
        _demo_state["initialized"] = True
        _demo_state["documents_processed"] = [os.path.basename(f.name if hasattr(f, 'name') else str(f)) for f in files]
        
        print("✓ RAG system ready for queries!")
        
        return f"✅ Successfully processed {len(files)} document(s) with {len(all_chunks)} chunks!\n\n📚 Documents loaded:\n" + "\n".join(f"  • {name}" for name in _demo_state["documents_processed"])
        
    except Exception as e:
        print(f"\n❌ Error during document processing: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


def initialize_rag_demo():
    """Initialize RAG system for UI demo (legacy function for backward compatibility)."""
    if _demo_state["initialized"]:
        return _demo_state["vector_store"], _demo_state["metadata_store"]
    
    SAVE_DIR = DOCUMENT_CONFIG["save_dir"]
    sample_path = DOCUMENT_CONFIG["sample_path"]
    
    # Check if user provided a default document path
    if not os.path.exists(sample_path):
        print("⚠️ No default document found. Please upload documents through the UI.")
        return None, None
    
    if check_if_embeddings_exist(SAVE_DIR):
        print("✓ Loading precomputed embeddings...")
        _demo_state["vector_store"] = VectorStore.load(SAVE_DIR, verbose=False)
        _demo_state["metadata_store"] = MetadataStore.load(
            os.path.join(SAVE_DIR, "metadata.pkl"), verbose=False
        )
    else:
        print("Processing document (this may take a few minutes)...")
        should_skip, vs, ms = skip_embedding_if_exists(SAVE_DIR, sample_path)
        
        if not should_skip:
            document_data = load_document(sample_path)
            is_pdf = isinstance(document_data, list)
            
            if is_pdf:
                chunks_data = chunk_pdf_pages(
                    document_data, 
                    chunk_size=DOCUMENT_CONFIG["chunk_size"],
                    overlap=DOCUMENT_CONFIG["chunk_overlap"]
                )
            else:
                chunks_data = chunk_document_semantic(
                    document_data, 
                    chunk_size=DOCUMENT_CONFIG["chunk_size"],
                    overlap=DOCUMENT_CONFIG["chunk_overlap"]
                )
            
            ms = MetadataStore()
            document_chunks = []
            for chunk_text, page_num in chunks_data:
                chunk = DocumentChunk(
                    text=chunk_text,
                    source=os.path.basename(sample_path),
                    page_number=page_num
                )
                ms.add_chunk(chunk)
                document_chunks.append(chunk)
            
            generate_embeddings(document_chunks, batch_size=100, use_optimizations=True)
            
            vs = VectorStore(EMBEDDING_DIM)
            embeddings = np.array([chunk.embedding for chunk in document_chunks])
            chunk_ids = [chunk.chunk_id for chunk in document_chunks]
            vs.add_vectors(embeddings, chunk_ids)
            
            os.makedirs(SAVE_DIR, exist_ok=True)
            vs.save(SAVE_DIR)
            ms.save(os.path.join(SAVE_DIR, "metadata.pkl"))
        else:
            vs = VectorStore.load(SAVE_DIR, verbose=False)
            ms = MetadataStore.load(os.path.join(SAVE_DIR, "metadata.pkl"), verbose=False)
        
        _demo_state["vector_store"] = vs
        _demo_state["metadata_store"] = ms
    
    _demo_state["initialized"] = True
    print("✓ RAG system ready for queries!")
    return _demo_state["vector_store"], _demo_state["metadata_store"]


def rag_query(user_query: str, model: str = None, top_k: int = None) -> str:
    """Simple interface for UI demos."""
    vs, ms = _demo_state["vector_store"], _demo_state["metadata_store"]
    if vs is None or ms is None:
        raise ValueError("RAG system not initialized. Call initialize_rag_demo() first.")
    
    result = execute_query(user_query, vs, ms, model_name=model, top_k=top_k, verbose=False)
    return result["formatted_answer"]


def rag_query_with_details(user_query: str, model: str = None, top_k: int = None) -> dict:
    """Advanced interface returning full result details."""
    vs, ms = _demo_state["vector_store"], _demo_state["metadata_store"]
    if vs is None or ms is None:
        raise ValueError("RAG system not initialized. Call initialize_rag_demo() first.")
    
    return execute_query(user_query, vs, ms, model_name=model, top_k=top_k, verbose=False)


# ===== GRADIO INTERFACE =====

def is_port_available(port: int) -> bool:
    """Check if a port is available for use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True
    except OSError:
        return False


def find_available_port(start_port: int = 7860, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")


def rag_demo_interface(query: str, top_k: int = 5, model: str = None) -> str:
    """Gradio-compatible RAG interface."""
    try:
        if not _demo_state["initialized"]:
            return "⚠️ Please upload and process documents first before querying!"
        
        if not query.strip():
            return "Please enter a question."
        
        if model is None or model.strip() == "":
            model = DEFAULT_OLLAMA_MODEL
        
        result = rag_query_with_details(query, model=model, top_k=top_k)
        return result["formatted_answer"]
    
    except ValueError as e:
        return f"❌ Error: {str(e)}\n\nPlease initialize the system first."
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"


def create_gradio_demo():
    """Create and configure Gradio interface for RAG system."""
    
    with gr.Blocks(
        title="📚 RAG Q&A Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gr-box { border-radius: 12px; }
        .gr-button { border-radius: 8px; }
        .gr-textbox { border-radius: 8px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📚 RAG Q&A Assistant
        
        **Step 1:** Upload your documents (PDF, TXT, MD, or DOCX)  
        **Step 2:** Click "Process Documents" to build the knowledge base  
        **Step 3:** Ask questions about your documents
        
        **Powered by:** FAISS + Ollama + HuggingFace Embeddings
        """)
        
        # Document Upload Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Upload Documents")
                file_upload = gr.File(
                    label="Upload Documents (PDF, TXT, MD, DOCX)",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx"],
                    type="filepath"
                )
                process_btn = gr.Button("🔄 Process Documents", variant="primary", size="lg")
                process_status = gr.Markdown("*No documents processed yet*")
        
        gr.Markdown("---")
        
        # Query Section
        gr.Markdown("### ❓ Ask Questions")
        
        # Query Section
        gr.Markdown("### ❓ Ask Questions")
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Your Question",
                    lines=3,
                    placeholder="e.g., What are the main topics discussed in the documents?",
                    interactive=True
                )
            
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    label="📊 Number of Sources",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    interactive=True
                )
                
                model_dropdown = gr.Textbox(
                    label="🤖 Model (optional)",
                    value=DEFAULT_OLLAMA_MODEL,
                    placeholder="Leave blank for default",
                    interactive=True
                )
        
        submit_btn = gr.Button("🔍 Search", variant="primary", scale=1)
        answer_output = gr.Markdown(
            label="📖 Answer with Sources",
            value="*Upload and process documents, then ask your questions here...*"
        )
        
        # Event handlers
        process_btn.click(
            fn=process_uploaded_documents,
            inputs=[file_upload],
            outputs=process_status
        )
        
        submit_btn.click(
            fn=rag_demo_interface,
            inputs=[query_input, top_k_slider, model_dropdown],
            outputs=answer_output
        )
        
        gr.Examples(
            examples=[
                ["What are the key concepts in this document?"],
                ["Summarize the main arguments."],
                ["What evidence is provided for the claims?"],
            ],
            inputs=query_input
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ How It Works
        1. **Upload**: Select one or more documents (PDF, TXT, MD, DOCX)
        2. **Process**: Click "Process Documents" to chunk and embed your documents
        3. **Query**: Ask questions - the system retrieves relevant chunks and generates answers
        4. **Citations**: Answers include [1], [2] citations linked to source documents
        
        ### 🚀 Performance
        - Document processing: ~1-5 minutes depending on size
        - Query response: <5 seconds
        - Supports multiple documents simultaneously
        - Completely private & offline (when using local models)
        """)
    
    return demo


def main():
    """Main entry point for the RAG application."""
    print("🚀 Starting RAG Terminal Application...")
    print("=" * 70)
    
    try:
        # Note: Documents will be uploaded through Gradio UI
        # No need to pre-initialize with default document
        print("📚 RAG system ready for document uploads...")
        print("✓ Upload documents through the Gradio interface\n")
        
        # Find available port
        preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
        
        if not is_port_available(preferred_port):
            print(f"⚠️  Port {preferred_port} is already in use. Finding available port...")
            available_port = find_available_port(preferred_port)
            print(f"✓ Using port {available_port} instead\n")
        else:
            available_port = preferred_port
            print(f"✓ Port {preferred_port} is available\n")
        
        # Create and launch Gradio interface
        gradio_demo = create_gradio_demo()
        
        print("🌐 Launching Gradio interface...")
        print("=" * 70)
        print(f"📖 Access the demo at: http://127.0.0.1:{available_port}")
        if share:
            print("🌍 Public link will be generated (share=True)")
        print("=" * 70)
        
        gradio_demo.launch(
            share=share,
            server_name=server_name,
            server_port=available_port,
        )
        
    except Exception as e:
        print(f"\n❌ Error launching RAG application: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that all required API keys are in .env file")
        print("  2. Verify Ollama is running (if using local models)")
        print("  3. Upload documents through the Gradio UI")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
