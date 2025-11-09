from dotenv import load_dotenv
import os, json
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
import faiss
from ollama import Client
import gradio as gr
import socket

# Handle different duckduckgo_search versions
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from duckduckgo_search import AsyncDDGS as DDGS
    except ImportError:
        DDGS = None
        print("⚠️  Warning: duckduckgo_search not available")


load_dotenv(override=False)

# print("Environment variables loaded from .env:")
# print(f"  ✓ OLLAMA_API_KEY present: {bool(os.environ.get('OLLAMA_API_KEY'))}")
# print(f"  ✓ HF_TOKEN present: {bool(os.environ.get('HF_TOKEN'))}")
# print(f"  ✓ OPENAI_API_KEY present: {bool(os.environ.get('OPENAI_API_KEY'))}")
# print(f"  ✓ VOYAGE_API_KEY present: {bool(os.environ.get('VOYAGE_API_KEY'))}")
# print("\nIf any required keys show False, check your .env file and re-run this cell.")


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
EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "huggingface")  # Options: "voyage", "openai", "huggingface", "fastembed"
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer ONLY using the provided sources.\n"
    "Format your response using proper markdown:\n"
    "- Use **bold** for emphasis\n"
    "- Use `code` for inline code\n"
    "- Use ```language blocks for code snippets\n"
    "- Use bullet points (- or *) for lists\n"
    "- Use numbered lists (1. 2. 3.) when appropriate\n"
    "- Cite evidence with bracketed indices like [1], [2]\n"
    "If unsure, say you don't know.\n"
    "Keep it concise: 3-6 sentences."
)

# Token limit configuration for LLM responses
# Base tokens for the answer, plus additional tokens per source
TOKEN_LIMIT_BASE = 512          # Minimum tokens for answer generation
TOKEN_LIMIT_PER_SOURCE = 200    # Additional tokens per source (for citations, context)
TOKEN_LIMIT_MAX = 2048          # Absolute maximum to prevent runaway responses

# Local LLMs via Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm2:360m")
OLLAMA_MODEL_CLOUD = os.getenv("OLLAMA_MODEL_CLOUD", "gpt-oss:20b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", None)  # Required for cloud models
client = Client(host=OLLAMA_BASE_URL)
# print(f"Using Ollama model: {OLLAMA_MODEL_CLOUD}")
# print(f"OLLAMA_API_KEY configured: {bool(OLLAMA_API_KEY)}")

# Initialize embedding client based on chosen method
if EMBEDDING_METHOD == "voyage" and VOYAGE_AVAILABLE and VOYAGE_API_KEY:
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    EMBEDDING_DIM = 1024  # voyage-2 dimension
    # print("Using Voyage AI embeddings (1024-dim)")
elif EMBEDDING_METHOD == "openai" and OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    EMBEDDING_DIM = 1536  # text-embedding-3-small dimension
    # print("Using OpenAI embeddings (1536-dim)")
elif EMBEDDING_METHOD == "huggingface" and HF_AVAILABLE and HF_TOKEN:
    hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
    HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIM = 768  # bge-base dimension
    # print(f"Using HuggingFace Inference API with {HF_MODEL} (768-dim)")
else:
    # Fallback to local fastembed
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    EMBEDDING_DIM = 384
    EMBEDDING_METHOD = "fastembed"
    # print("Using local FastEmbed (384-dim) - Set API keys for faster embeddings")
    # print("  Options: VOYAGE_API_KEY, OPENAI_API_KEY, or HF_TOKEN")

# print("Ollama client ready; embedding system initialized.")


# ===== CENTRALIZED CONFIGURATION =====

# Document & Processing Config
DOCUMENT_CONFIG = {
    "sample_path": "/home/agupta/Documents/Books/Atomic_Habits_James_Clear.pdf",
    "save_dir": "./rag_data",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
}

# Model Config
MODEL_CONFIG = {
    "ollama_local": os.getenv("OLLAMA_MODEL", "smollm2:360m"),
    "ollama_cloud": os.getenv("OLLAMA_MODEL_CLOUD", "gpt-oss:20b-cloud"),
    "api_key": os.getenv("OLLAMA_API_KEY", None),
}

# LLM Response Token Limits
# These control how long the LLM can generate responses
# More sources = more tokens allowed (for proper citations)
# Formula: max_tokens = min(BASE + (num_sources * PER_SOURCE), MAX)
TOKEN_CONFIG = {
    "base": 512,           # Minimum tokens for answer generation (was 256, now 512)
    "per_source": 200,     # Additional tokens per source for citations (~3-5 per citation)
    "max": 2048,           # Absolute maximum to prevent runaway responses
}

# Select model: prefer cloud if API key available, else local
DEFAULT_OLLAMA_MODEL = MODEL_CONFIG["ollama_cloud"] if MODEL_CONFIG["api_key"] else MODEL_CONFIG["ollama_local"]

# Global state for UI demos (initialized once)
_demo_state = {
    "vector_store": None,
    "metadata_store": None,
    "initialized": False,
}


# ===== EMBEDDING OPTIMIZATION LAYER =====

class EmbeddingCache:
    """
    File-based cache for embeddings using content hash as key.
    Dramatically speeds up repeated queries on same documents.
    """
    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}  # In-memory cache for current session
    
    def _hash_text(self, text: str) -> str:
        """Generate hash key from text content."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> np.ndarray:
        """Retrieve embedding from cache if exists."""
        key = self._hash_text(text)
        
        # Check memory cache first (faster)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file)
            self.memory_cache[key] = embedding  # Cache in memory for this session
            return embedding
        
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in both memory and disk cache."""
        key = self._hash_text(text)
        self.memory_cache[key] = embedding
        
        # Save to disk
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)


class ParallelHFEmbedder:
    """
    HuggingFace embeddings with parallel processing (4 workers).
    ~5-10x faster than sequential calls for many texts.
    """
    def __init__(self, model: str, api_key: str, num_workers: int = 4):
        self.model = model
        self.api_key = api_key
        self.num_workers = num_workers
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts in parallel."""
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
                    results[idx] = np.zeros(768, dtype="float32")  # Fallback
        
        # Use thread pool for parallel requests
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for idx, text in enumerate(texts):
                future = executor.submit(embed_single, idx, text)
                futures.append(future)
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        return np.array([r for r in results if r is not None], dtype="float32")


class LocalFastEmbedder:
    """
    Local sentence-transformers embeddings with batch processing.
    Faster than API calls but requires local GPU/CPU.
    ~30-60 seconds for 660 chunks on CPU.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            print("⚠️  sentence-transformers not available. Install with: pip install sentence-transformers")
            self.available = False
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts using local sentence-transformers."""
        if not self.available:
            return None
        
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings, dtype="float32")

print("✓ Embedding optimization layer loaded (caching, parallel, local support)")


class DocumentChunk:
    def __init__(self, text: str, source: str, page_number: int = None, chunk_id: str = None, metadata: dict = None):
        self.chunk_id = chunk_id if chunk_id is not None else str(uuid.uuid4())
        self.text = text
        self.source = source
        self.page_number = page_number
        self.embedding = None  # To be filled later
        self.metadata = metadata if metadata is not None else {}  # Store additional metadata (e.g., for web sources)


class MetadataStore:
    def __init__(self):
        self.chunks = {}  # Stores chunks with chunk_id as key

    def add_chunk(self, chunk: DocumentChunk):
        self.chunks[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> DocumentChunk:
        return self.chunks.get(chunk_id)

    def get_all_chunks(self) -> list[DocumentChunk]:
        return list(self.chunks.values())

    def index_chunks(self):
        """
        Placeholder for more sophisticated indexing.
        Currently chunks are indexed by chunk_id in the dictionary.
        """
        print(f"Indexed {len(self.chunks)} chunks in metadata store.")

    def save(self, save_path: str):
        """Saves metadata store to disk."""
        with open(save_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Metadata store saved to {save_path}")

    @classmethod
    def load(cls, save_path: str, verbose: bool = True):
        """Loads metadata store from disk.
        
        Args:
            save_path: Path to metadata file
            verbose: If True, print loading messages. If False, load silently.
        """
        instance = cls()
        with open(save_path, 'rb') as f:
            instance.chunks = pickle.load(f)
        
        if verbose:
            print(f"Metadata store loaded from {save_path} ({len(instance.chunks)} chunks)")
        return instance


class VectorStore:
    def __init__(self, dimension: int):
        """
        Initialize a Faiss index using Inner Product (cosine similarity after normalization).
        IndexFlatIP is better for embeddings than IndexFlatL2.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity with normalized vectors
        self.chunk_ids = []  # Maps Faiss index to chunk IDs

    def add_vectors(self, embeddings: np.ndarray, chunk_ids: list[str]):
        """
        Adds embeddings to the Faiss index after L2 normalization.
        Normalization converts inner product to cosine similarity.
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        print(f"Added {len(chunk_ids)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Searches for the nearest neighbors using cosine similarity.
        Returns list of chunk IDs sorted by relevance.
        """
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search (higher scores = more similar for IP)
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        # Retrieve corresponding chunk IDs
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.chunk_ids):
                results.append(self.chunk_ids[i])
        return results

    def save(self, save_dir: str):
        """Saves the vector store to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save chunk IDs and metadata
        metadata_path = os.path.join(save_dir, "chunk_ids.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunk_ids': self.chunk_ids,
                'dimension': self.dimension
            }, f)
        
        print(f"Vector store saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, verbose: bool = True):
        """Loads the vector store from disk.
        
        Args:
            save_dir: Directory containing saved vector store
            verbose: If True, print loading messages. If False, load silently.
        """
        index_path = os.path.join(save_dir, "faiss_index.bin")
        metadata_path = os.path.join(save_dir, "chunk_ids.pkl")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance and load index
        instance = cls(metadata['dimension'])
        instance.index = faiss.read_index(index_path)
        instance.chunk_ids = metadata['chunk_ids']
        
        if verbose:
            print(f"Vector store loaded from {save_dir} ({instance.index.ntotal} vectors)")
        return instance
    

# ===== UTF-8 ENCODING FIX HELPER =====

def fix_utf8_encoding(text: str) -> str:
    """
    Fix UTF-8 encoding issues in text (e.g., \\u2019 → ', \\u2014 → —).
    Handles Unicode escape sequences and ensures proper markdown rendering.
    """
    if not text:
        return text
    
    try:
        # First, try to encode as raw bytes and decode properly
        # This handles cases where text contains literal backslash-u sequences
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        # Replace common Unicode escapes manually (fallback for edge cases)
        replacements = {
            r'\u2019': "'",      # Right single quotation mark
            r'\u2018': "'",      # Left single quotation mark
            r'\u201c': '"',      # Left double quotation mark
            r'\u201d': '"',      # Right double quotation mark
            r'\u2014': '—',      # Em dash
            r'\u2013': '–',      # En dash
            r'\u2026': '...',    # Ellipsis
            r'\u00a0': ' ',      # Non-breaking space
        }
        
        for escape, replacement in replacements.items():
            text = text.replace(escape, replacement)
        
        return text
    except Exception as e:
        print(f"Warning: UTF-8 fix error: {e}")
        return text
    

# # ===== TEST UTF-8 ENCODING FIX =====

# print("=" * 70)
# print("🧪 TESTING UTF-8 ENCODING FIX")
# print("=" * 70)

# # Test cases with common problematic Unicode sequences
# test_cases = [
#     ("It\\u2019s a great day!", "It's a great day!"),
#     ("The book\\u2014by James Clear\\u2014is excellent.", "The book—by James Clear—is excellent."),
#     ("He said, \\u201cHello!\\u201d", 'He said, "Hello!"'),
#     ("Wait\\u2026 what?", "Wait... what?"),
# ]

# print("\n📝 Testing UTF-8 encoding fixes:\n")

# for input_text, expected_output in test_cases:
#     fixed_text = fix_utf8_encoding(input_text)
#     status = "✅" if fixed_text == expected_output else "❌"
#     print(f"{status} Input:    {repr(input_text)}")
#     print(f"   Output:   {repr(fixed_text)}")
#     print(f"   Expected: {repr(expected_output)}")
#     print()

# print("=" * 70)
# print("✓ UTF-8 encoding fix is working correctly!")
# print("=" * 70)


def embed_texts_optimized(texts: list[str], batch_size: int = 100, 
                         use_cache: bool = True, 
                         use_parallel: bool = True,
                         use_local_fast: bool = False) -> np.ndarray:
    """
    Optimized embedding generation with caching, parallel processing, and local fallback.
    
    Optimizations:
    - Caching: Reuse embeddings for texts we've seen before (10x+ speedup)
    - Parallel: Multi-threaded API calls for HuggingFace (5-10x speedup)
    - Local fast: Sentence-transformers batch processing (fastest local option)
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for API calls (default: 100)
        use_cache: Enable file-based caching (default: True)
        use_parallel: Use parallel HF API calls (default: True)
        use_local_fast: Use local sentence-transformers instead of API (default: False)
    
    Returns:
        numpy array of embeddings (shape: [len(texts), embedding_dim])
    """
    
    # Initialize cache if enabled
    cache = EmbeddingCache() if use_cache else None
    
    # Check cache for existing embeddings
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
            print(f"✓ Found {len(cached_embeddings)} cached embeddings (skipping {len(cached_embeddings)} API calls)")
    else:
        uncached_texts = texts
        uncached_indices = list(range(len(texts)))
    
    # If everything is cached, return early
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
    
    # Embed uncached texts
    if use_parallel and EMBEDDING_METHOD == "huggingface" and HF_AVAILABLE and HF_TOKEN:
        print(f"Embedding {len(uncached_texts)} texts using parallel HuggingFace (4 workers)...")
        parallel_embedder = ParallelHFEmbedder(HF_MODEL, HF_TOKEN, num_workers=4)
        new_embeddings = parallel_embedder.embed_batch(uncached_texts)
    else:
        # Sequential embedding (with all methods supported)
        print(f"Embedding {len(uncached_texts)} texts...")
        new_embeddings = []
        
        for i in tqdm(range(0, len(uncached_texts), batch_size), desc="Embedding", disable=len(uncached_texts) < batch_size):
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
    
    # Merge cached and new embeddings in correct order
    all_embeddings = np.zeros((len(texts), new_embeddings.shape[1]), dtype="float32")
    
    # Place cached embeddings
    for i, emb in cached_embeddings.items():
        all_embeddings[i] = emb
    
    # Place new embeddings
    for idx, emb in zip(uncached_indices, new_embeddings):
        all_embeddings[idx] = emb
    
    return all_embeddings

# print("✓ Optimized embedding function loaded (embed_texts_optimized)")


def load_document(file_path: str):
    """Loads content from various document types."""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        return load_pdf(file_path)
    elif file_extension == '.md':
        return load_text(file_path)
    elif file_extension == '.txt':
        return load_text(file_path)
    elif file_extension == '.docx':
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def load_pdf(file_path: str) -> List[Tuple[str, int]]:
    """
    Loads text from a PDF file using PyMuPDF (faster than pypdf).
    Returns list of (text, page_number) tuples to preserve page information.
    """
    try:
        doc = fitz.open(file_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean up text: remove excessive whitespace
            text = ' '.join(text.split())
            
            if text.strip():  # Only add non-empty pages
                pages_data.append((text, page_num + 1))  # 1-indexed page numbers
        
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


def load_text_file(file_obj) -> str:
    """
    Loads text from a file object (e.g., from Gradio upload).
    Works with both file paths and file-like objects.
    """
    try:
        # If it's a file path (string)
        if isinstance(file_obj, str):
            return load_text(file_obj)
        
        # If it's a file object with .read() method
        if hasattr(file_obj, 'read'):
            content = file_obj.read()
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='ignore')
            return content
        
        # If it's a temporary file path (Gradio creates temp files)
        if hasattr(file_obj, 'name'):
            return load_text(file_obj.name)
        
        raise ValueError(f"Unsupported file object type: {type(file_obj)}")
    except Exception as e:
        print(f"Error loading text file: {e}")
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
    

def chunk_document_semantic(
    document_text: str, 
    chunk_size: int = 1000, 
    overlap: int = 200,
    page_number: int = None
) -> List[Tuple[str, int]]:
    """
    Splits a document into semantic chunks that respect sentence boundaries.
    
    Args:
        document_text: The text to chunk
        chunk_size: Target size for each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        page_number: Page number for this text (if from PDF)
    
    Returns:
        List of (chunk_text, page_number) tuples
    """
    # Split into sentences (handles common sentence endings)
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(document_text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence keeps us under chunk_size, add it
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            # Save current chunk if it's not empty
            if current_chunk.strip():
                chunks.append((current_chunk.strip(), page_number))
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                # Take last 'overlap' characters, but try to start at sentence boundary
                overlap_text = current_chunk[-overlap:]
                # Find the first sentence start in the overlap
                first_sentence_start = overlap_text.find('. ')
                if first_sentence_start != -1:
                    overlap_text = overlap_text[first_sentence_start + 2:]
                current_chunk = overlap_text + sentence + " "
            else:
                current_chunk = sentence + " "
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append((current_chunk.strip(), page_number))
    
    return chunks


def chunk_pdf_pages(pages_data: List[Tuple[str, int]], chunk_size: int = 1000, overlap: int = 200):
    """
    Chunks PDF pages while preserving page number information.
    
    Args:
        pages_data: List of (text, page_number) tuples from load_pdf
        chunk_size: Target chunk size
        overlap: Overlap between chunks
    
    Returns:
        List of (chunk_text, page_number) tuples
    """
    all_chunks = []
    
    for page_text, page_num in tqdm(pages_data, desc="Chunking pages"):
        page_chunks = chunk_document_semantic(page_text, chunk_size, overlap, page_num)
        all_chunks.extend(page_chunks)
    
    return all_chunks


def chunk_document_manual(document_text: str, max_chars_per_chunk: int = 1000):
    """
    DEPRECATED: Simple character-based chunking (kept for backward compatibility).
    Use chunk_document_semantic() instead for better results.
    """
    print("⚠️  Warning: Using deprecated simple chunking. Consider using chunk_document_semantic()")
    chunks = []
    for i in range(0, len(document_text), max_chars_per_chunk):
        chunks.append(document_text[i:i + max_chars_per_chunk])
    return chunks


def generate_embeddings(chunks: list[DocumentChunk], batch_size: int = 100, use_optimizations: bool = True):
    """
    Generates embeddings for chunks using configured method with optimizations.
    
    Optimizations include:
    - Caching: File-based cache for repeated texts (10x+ speedup)
    - Parallelization: Thread pool for HuggingFace API calls (5-10x speedup)
    - Local fast: Sentence-transformers batch processing (fastest, no API)
    
    Args:
        chunks: List of DocumentChunk objects to embed
        batch_size: Batch size for processing
        use_optimizations: Whether to enable caching, parallel, and local embeddings
    """
    texts = [chunk.text for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks using optimized {EMBEDDING_METHOD}...")
    print(f"  Options: caching (10x+), parallel HF (5-10x), local fast embedding")
    embeddings = embed_texts_optimized(
        texts, 
        batch_size=batch_size,
        use_cache=use_optimizations,           # Enable caching
        use_parallel=use_optimizations,        # Enable parallelization
        use_local_fast=False                   # Keep False for API methods, True for local
    )
    
    # Assign embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    
    print(f"✓ Generated {len(embeddings)} embeddings ({EMBEDDING_DIM}-dimensional)")


def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Simple embedding function for queries (single text).
    Uses the same method as document embeddings but optimized for quick queries.
    """
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


def retrieve_documents(query: str, vector_store: VectorStore, metadata_store: MetadataStore, k: int = 5):
    """
    Retrieves the top K most relevant document chunks for a given query.
    """
    # 1. Generate embedding for the query
    qv = embed_texts([query])
    # 2. Search the vector store for similar chunks
    retrieved_chunk_ids = vector_store.search(qv[0], k=k)
    # 3. Retrieve the actual chunk objects from the metadata store
    retrieved_chunks = [metadata_store.get_chunk(chunk_id) for chunk_id in retrieved_chunk_ids if metadata_store.get_chunk(chunk_id) is not None]
    return retrieved_chunks


def search_web(query: str, num_results: int = 5) -> list[dict]:
    """Search the web and return results."""
    if DDGS is None:
        print("⚠️  Web search unavailable: duckduckgo_search not installed")
        return []
    
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region='us-en', max_results=num_results):
                results.append({
                    "text": r.get('body', ''),
                    "source": r.get('title', ''),
                    "url": r.get('link', r.get('url', r.get('href', ''))),
                    "page": None  # No page number for web results
                })
    except Exception as e:
        print(f"⚠️  Web search failed: {e}")
        return []
    return results


def retrieve_documents_hybrid(query: str, vector_store, metadata_store, 
                              k_local: int = 3, k_web: int = 2):
    """Retrieve from both local docs and web."""
    
    # Local retrieval (existing code)
    local_chunks = retrieve_documents(query, vector_store, metadata_store, k=k_local)
    
    # Web search
    web_results = search_web(query, num_results=k_web)
    
    # Convert web results to DocumentChunk format
    web_chunks = [
        DocumentChunk(
            text=r['text'],
            source=r['source'],
            page_number=None,
            metadata={'url': r['url'], 'type': 'web'}
        )
        for r in web_results
    ]
    
    # Merge (local results ranked higher)
    return local_chunks + web_chunks


try:
    client
except NameError:
    client = Client(host=OLLAMA_BASE_URL)


def generate_answer(query: str, retrieved_chunks: list, model_name: str = None, stream: bool = False):
    """
    Generate an answer from Ollama or Ollama cloud, with defensive checks.

    - Handles None chunk.text
    - Validates cloud API key before using cloud model
    - Uses == for string comparison
    - Supports stream flag if the client and model support streaming
    - Dynamically adjusts token limit based on number of sources
    """
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL", "smollm2:360m")

    # Build context lines safely
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        src = getattr(chunk, "source", f"doc_{i}")
        # ensure chunk.text is a string
        chunk_text_raw = getattr(chunk, "text", "")
        if chunk_text_raw is None:
            chunk_text = ""
        else:
            chunk_text = str(chunk_text_raw)
        # trim to avoid sending too much
        context_lines.append(f"[{i}] {src}:\n{chunk_text[:800]}")

    user = f"Question: {query}\n\nSources:\n" + "\n\n".join(context_lines) + "\n\nAnswer:"

    # Calculate dynamic token limit based on number of sources
    num_sources = len(retrieved_chunks)
    num_predict = min(
        TOKEN_LIMIT_BASE + (num_sources * TOKEN_LIMIT_PER_SOURCE),
        TOKEN_LIMIT_MAX
    )
    
    print(f"[LLM Config] Sources: {num_sources} | Max tokens: {num_predict}")

    # Decide which client to use
    # If using cloud model name, ensure API key exists
    if model_name == OLLAMA_MODEL_CLOUD:
        if not OLLAMA_API_KEY:
            raise ValueError("OLLAMA_API_KEY is not set but cloud model was requested. Set OLLAMA_API_KEY or use the local model.")
        # create a client configured for cloud (keep default client for local)
        cloud_client = Client(host="https://ollama.com", headers={"Authorization": "Bearer " + OLLAMA_API_KEY})
        chosen_client = cloud_client
    else:
        chosen_client = client

    # Prepare messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    # Try streaming if requested and the client supports it
    try:
        if stream:
            # Some clients support streaming via `stream=True` or a stream() method
            # We'll attempt a streaming call but fall back to non-streaming safely
            resp_stream = chosen_client.chat(model=model_name, messages=messages, stream=True, options={"temperature": 0.2, "num_predict": num_predict})
            # If resp_stream is an iterator of chunks/dicts, iterate and concatenate
            full_text = ""
            try:
                for part in resp_stream:
                    # Some streaming responses yield dicts with 'message' -> 'content'
                    if isinstance(part, dict) and "message" in part and isinstance(part["message"], dict):
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
                # Not iterable; fall back
                pass

        # Non-streaming call (or fallback)
        resp = chosen_client.chat(model=model_name, messages=messages, options={"temperature": 0.2, "num_predict": num_predict})
        # resp may be a dict like {'message': {'content': '...'}}
        if isinstance(resp, dict):
            message = resp.get("message")
            if isinstance(message, dict):
                return message.get("content", "").strip()
            # sometimes response might be {'content': '...'}
            return resp.get("content", "").strip()

        # If resp is a string, return it
        if isinstance(resp, str):
            return resp.strip()

        # Unknown shape
        return str(resp)

    except Exception as e:
        # Don't raise raw exceptions to the user; return a helpful message
        return f"Error generating answer: {e}"


def format_answer_with_sources(answer_text: str, retrieved_chunks: list) -> str:
    """
    Format answer with source citations [1], [2], etc. and source details below.
    
    Example output:
    ---
    The Four Laws of Behavior Change are cue, craving, response, and reward [1]. 
    Each law builds on the previous one to form a complete habit loop [2].
    
    📚 SOURCES:
    [1] Atomic_Habits_James_Clear.pdf (Page 42):
        "The four laws of behavior change are the cue, the craving, the response..."
    
    [2] Atomic_Habits_James_Clear.pdf (Page 45):
        "Together, these four elements form the habit loop which is the..."
    ---
    """
    if not retrieved_chunks or not answer_text.strip():
        return answer_text
    
    # Format source information
    sources_section = "\n📚 SOURCES:\n"
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        source_name = getattr(chunk, "source", f"Document {i}")
        page_num = getattr(chunk, "page_number", None)
        chunk_text = getattr(chunk, "text", "")
        
        # Safely handle None text
        if chunk_text is None:
            chunk_text = "(No text content)"
        else:
            chunk_text = str(chunk_text)
        
        # Truncate text snippet to 200 characters for readability
        snippet = chunk_text[:200].replace("\n", " ")
        if len(chunk_text) > 200:
            snippet += "..."
        
        # Format source entry
        metadata = getattr(chunk, "metadata", {})
        if isinstance(metadata, dict) and metadata.get('type') == 'web':
            sources_section += f"\n[{i}] 🌐 {source_name}\n"
            sources_section += f"    {metadata.get('url', '')}\n"
        else:
            page_info = f" (Page {page_num})" if page_num else ""
            sources_section += f"\n[{i}] {source_name}{page_info}:\n"
            sources_section += f'    "{snippet}"\n'
    
    # Combine answer with sources
    formatted = f"{answer_text}\n{sources_section}"
    return formatted

print("✓ Formatted answer function loaded (with [1], [2] citations and sources)")


# ===== REUSABLE HELPER FUNCTIONS =====

def extract_content(resp):
    """
    Extract answer text from various response formats (dict, string, JSON).
    Handles responses from local Ollama, cloud Ollama, and API models.
    Applies UTF-8 encoding fix to ensure proper character display.
    """
    if resp is None:
        return ""
    
    # Direct dict-like response
    if isinstance(resp, dict):
        msg = resp.get("message") or resp.get("content")
        if isinstance(msg, dict):
            content = msg.get("content", "").strip()
            return fix_utf8_encoding(content)
        if isinstance(msg, str):
            return fix_utf8_encoding(msg.strip())
        return fix_utf8_encoding(str(resp).strip())
    
    # If it's not a string, stringify it
    if not isinstance(resp, str):
        return fix_utf8_encoding(str(resp).strip())
    
    s = resp
    
    # Try JSON parse if possible
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            msg = parsed.get("message")
            if isinstance(msg, dict):
                content = msg.get("content", "").strip()
                return fix_utf8_encoding(content)
    except Exception:
        pass
    
    # Regex: message=Message(... content='...') or content='...' patterns
    m = re.search(r"message=Message\([^)]*content=(?P<q>['\"])(?P<content>.*?)(?P=q)", s, re.DOTALL)
    if m:
        return fix_utf8_encoding(m.group("content").strip())
    
    m2 = re.search(r"content=(?P<q>['\"])(?P<content>.*?)(?P=q)", s, re.DOTALL)
    if m2:
        return fix_utf8_encoding(m2.group("content").strip())
    
    # Fallback: return the original string with UTF-8 fix
    return fix_utf8_encoding(s.strip())


def execute_query(query: str, vector_store: VectorStore, metadata_store: MetadataStore, 
                 model_name: str = None, top_k: int = None, verbose: bool = True) -> dict:
    """
    End-to-end query execution: retrieve documents → generate answer → format with sources.
    
    Args:
        query: User's question
        vector_store: Loaded FAISS vector store
        metadata_store: Loaded metadata store
        model_name: LLM model to use (defaults to DEFAULT_OLLAMA_MODEL)
        top_k: Number of chunks to retrieve (defaults to DOCUMENT_CONFIG["top_k"])
        verbose: Print progress messages
    
    Returns:
        dict with keys: "query", "answer", "formatted_answer", "retrieved_chunks"
    """
    if model_name is None:
        model_name = DEFAULT_OLLAMA_MODEL
    if top_k is None:
        top_k = DOCUMENT_CONFIG["top_k"]
    
    if verbose:
        print(f"📖 Query: {query}")
    
    # Retrieve documents
    retrieved_chunks = retrieve_documents_hybrid(query, vector_store, metadata_store, k_local=top_k//2, k_web=top_k//2) # retrieve_documents(query, vector_store, metadata_store, k=top_k)
    if verbose:
        print(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")
    
    # Generate answer
    if verbose:
        print(f"Generating answer with model: {model_name}")
    
    raw_response = generate_answer(query, retrieved_chunks, model_name=model_name, stream=False)
    answer_text = extract_content(raw_response)
    
    # Format with sources
    formatted_output = format_answer_with_sources(answer_text, retrieved_chunks)
    
    return {
        "query": query,
        "answer": answer_text,
        "formatted_answer": formatted_output,
        "retrieved_chunks": retrieved_chunks,
        "model": model_name
    }


def display_result(result: dict):
    """Pretty print query result with answer and sources."""
    print("\n" + "=" * 70)
    print("ANSWER WITH FORMATTED SOURCES:")
    print("=" * 70)
    print(result["formatted_answer"])
    print("=" * 70)

# print("✓ Reusable helper functions loaded (extract_content, execute_query, display_result)")
# print("✓ UTF-8 encoding fix applied to all LLM responses")
# print("✓ Configuration centralized in DOCUMENT_CONFIG, MODEL_CONFIG, and TOKEN_CONFIG")


def check_if_embeddings_exist(save_dir: str) -> bool:
    """
    Check if embeddings and metadata already exist from a previous run.
    Returns True if all necessary files are present and valid.
    """
    import os
    
    required_files = [
        os.path.join(save_dir, "faiss_index.bin"),
        os.path.join(save_dir, "chunk_ids.pkl"),
        os.path.join(save_dir, "metadata.pkl")
    ]
    
    # Check if all files exist
    all_exist = all(os.path.exists(f) for f in required_files)
    
    if all_exist:
        try:
            # Try to load to verify integrity (silently)
            test_vs = VectorStore.load(save_dir, verbose=False)
            test_ms = MetadataStore.load(os.path.join(save_dir, "metadata.pkl"), verbose=False)
            
            # Check if they have content
            if test_vs.index.ntotal > 0 and len(test_ms.get_all_chunks()) > 0:
                return True
        except Exception as e:
            print(f"⚠️  Embeddings exist but failed to load: {e}")
    
    return False


def skip_embedding_if_exists(save_dir: str, sample_document_path: str):
    """
    Smart pipeline state manager:
    - If embeddings exist: Load them and skip document processing
    - If embeddings don't exist: Process document normally
    
    Returns: (should_skip_embedding, vector_store, metadata_store)
    """
    if check_if_embeddings_exist(save_dir):
        print("\n" + "=" * 70)
        print("⚡ FOUND PRECOMPUTED EMBEDDINGS - SKIPPING REDUNDANT API CALLS")
        print("=" * 70)
        print(f"✓ Embeddings already exist in {save_dir}")
        print(f"✓ Skipping: document loading, chunking, and embedding generation")
        print(f"✓ Directly loading from disk...\n")
        
        # Load silently since we're providing our own status messages
        vector_store = VectorStore.load(save_dir, verbose=False)
        metadata_store = MetadataStore.load(os.path.join(save_dir, "metadata.pkl"), verbose=False)
        
        print(f"✓ Loaded {vector_store.index.ntotal} precomputed embeddings")
        print(f"✓ Loaded {len(metadata_store.get_all_chunks())} document chunks")
        print(f"✓ Ready for querying!\n")
        
        return True, vector_store, metadata_store
    else:
        print(f"\n📄 No precomputed embeddings found. Processing document: {os.path.basename(sample_document_path)}")
        return False, None, None

print("✓ Smart embedding state checker loaded (avoids redundant API calls)")


# # ===== INTELLIGENT PIPELINE EXECUTION (WITH SMART SKIP) =====

# print("=" * 70)
# print("RAG PIPELINE - Smart Embedding Reuse (Avoids Redundant API Calls)")
# print("=" * 70)

# # Unpack configuration
# sample_document_path = DOCUMENT_CONFIG["sample_path"]
# SAVE_DIR = DOCUMENT_CONFIG["save_dir"]
# CHUNK_SIZE = DOCUMENT_CONFIG["chunk_size"]
# CHUNK_OVERLAP = DOCUMENT_CONFIG["chunk_overlap"]

# # Check if embeddings already exist from previous run
# should_skip_embedding, vector_store, metadata_store = skip_embedding_if_exists(SAVE_DIR, sample_document_path)

# if should_skip_embedding:
#     # Embeddings exist - skip document processing and go straight to querying
#     print("[1/2] ✓ SKIPPED - Using precomputed embeddings")
#     print("[2/2] Testing retrieval & answer generation with formatted sources...")
# else:
#     # Embeddings don't exist - process document normally
    
#     # Step 1: Load Document
#     print("\n[1/7] Loading document...")
#     try:
#         document_data = load_document(sample_document_path)
#         if isinstance(document_data, list):  # PDF returns list of (text, page_num)
#             print(f"✓ Loaded PDF with {len(document_data)} pages")
#             is_pdf = True
#         elif isinstance(document_data, str):  # Text/DOCX returns string
#             print(f"✓ Loaded document ({len(document_data)} characters)")
#             is_pdf = False
#             document_data = [(document_data, None)]  # Convert to same format
#         else:
#             raise ValueError("Failed to load document")
#     except Exception as e:
#         print(f"✗ Error loading document: {e}")
#         raise

#     # Step 2: Chunk Document
#     print(f"\n[2/7] Chunking document (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
#     if is_pdf:
#         chunks_data = chunk_pdf_pages(document_data, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
#     else:
#         chunks_data = chunk_document_semantic(document_data[0][0], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

#     print(f"✓ Created {len(chunks_data)} semantic chunks")

#     # Step 3: Create Metadata Store
#     print("\n[3/7] Building metadata store...")
#     metadata_store = MetadataStore()
#     document_chunks = []

#     for chunk_text, page_num in chunks_data:
#         chunk = DocumentChunk(
#             text=chunk_text,
#             source=os.path.basename(sample_document_path),
#             page_number=page_num
#         )
#         metadata_store.add_chunk(chunk)
#         document_chunks.append(chunk)

#     metadata_store.index_chunks()

#     # Step 4: Generate Embeddings (WITH OPTIMIZATIONS)
#     print(f"\n[4/7] Generating embeddings using {EMBEDDING_METHOD}...")
#     print("      ⚡ Enabling: caching, parallel processing, and batch optimization")
#     generate_embeddings(document_chunks, batch_size=100, use_optimizations=True)

#     # Step 5: Build Vector Store
#     print("\n[5/7] Building vector store with cosine similarity...")
#     vector_store = VectorStore(EMBEDDING_DIM)

#     embeddings = np.array([chunk.embedding for chunk in document_chunks])
#     chunk_ids = [chunk.chunk_id for chunk in document_chunks]

#     vector_store.add_vectors(embeddings, chunk_ids)

#     # Step 6: Save Everything to Disk
#     print(f"\n[6/7] Saving to disk ({SAVE_DIR})...")
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     vector_store.save(SAVE_DIR)
#     metadata_store.save(os.path.join(SAVE_DIR, "metadata.pkl"))
    
#     print(f"✓ Embeddings saved! Next run will reuse them automatically.")
#     print(f"\n[7/7] Testing retrieval & answer generation with formatted sources...")

# # ===== COMMON PATH (for both first run and subsequent runs) =====

# query = "What are the Four Laws of Behavior Change?"
# result = execute_query(query, vector_store, metadata_store, verbose=True)
# display_result(result)

# if should_skip_embedding:
#     print(f"\n✓ Query complete! (Using cached embeddings)")
#     print(f"  To regenerate embeddings: delete {SAVE_DIR}/ or set should_skip_embedding=False")
# else:
#     print(f"\n✓ Pipeline complete! Data saved to {SAVE_DIR}")
#     print(f"  Embedding cache stored in: .embedding_cache/")
#     print(f"  Next run will reuse embeddings automatically!")
#     print(f"  To reload manually: vector_store = VectorStore.load('{SAVE_DIR}')")
#     print(f"                      metadata_store = MetadataStore.load('{SAVE_DIR}/metadata.pkl')")


# # ===== DEMO PREPARATION: UI-Ready Functions =====

# def try_load_existing_embeddings():
#     """
#     Attempt to load precomputed embeddings from disk.
#     Returns (success: bool, vector_store, metadata_store)
#     """
#     SAVE_DIR = DOCUMENT_CONFIG["save_dir"]
    
#     try:
#         if check_if_embeddings_exist(SAVE_DIR):
#             print("✓ Loading precomputed embeddings from disk...")
#             vs = VectorStore.load(SAVE_DIR, verbose=False)
#             ms = MetadataStore.load(
#                 os.path.join(SAVE_DIR, "metadata.pkl"), verbose=False
#             )
#             return True, vs, ms
#     except Exception as e:
#         print(f"⚠️  Could not load embeddings: {e}")
    
#     return False, None, None


# def initialize_rag_demo():
#     """
#     Initialize RAG system for UI demo.
#     Loads precomputed embeddings if available, otherwise waits for user to upload documents.
#     Safe to call multiple times - only initializes once.
#     """
#     if _demo_state["initialized"]:
#         return _demo_state["vector_store"], _demo_state["metadata_store"]
    
#     # Try to load existing embeddings
#     success, vs, ms = try_load_existing_embeddings()
    
#     if success:
#         _demo_state["vector_store"] = vs
#         _demo_state["metadata_store"] = ms
#         _demo_state["initialized"] = True
#         print("✓ RAG system ready to use!")
#         return vs, ms
    
#     else:
#         # No precomputed embeddings - wait for user to upload documents
#         print("\n📚 RAG System Ready for Document Upload")
#         print("   Waiting for user to upload documents via the UI...")
#         print("   No local documents to process automatically.")
#         print("\n   To get started:")
#         print("   1. Open the '📤 Upload Documents' tab")
#         print("   2. Upload your PDF/DOCX/TXT files")
#         print("   3. Click 'Process Documents'")
#         print("   4. Switch to '🔍 Ask Questions' tab to query\n")
        
#         return None, None
