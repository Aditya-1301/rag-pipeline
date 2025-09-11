from dataclasses import dataclass
import os

@dataclass
class Settings:
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "fastembed")
    FASTEMBED_MODEL: str = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING: str = os.getenv("OPENAI_EMBEDDING", "text-embedding-3-small")

    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "350"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    DATA_DIR: str = os.getenv("DATA_DIR", ".rag")
    INDEX_PATH: str = os.getenv("INDEX_PATH", ".rag/faiss.index")
    DOCSTORE_PATH: str = os.getenv("DOCSTORE_PATH", ".rag/docstore.jsonl")

settings = Settings()
