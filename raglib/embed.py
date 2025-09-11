# raglib/embed.py
from typing import List
import numpy as np
from .config import settings

class Embedder:
    def __init__(self):
        self.backend = settings.EMBEDDING_BACKEND
        if self.backend == "fastembed":
            from fastembed import TextEmbedding
            # Force CPU provider to avoid any ambiguity
            self.model = TextEmbedding(model_name=settings.FASTEMBED_MODEL)
            self._dim = None
        elif self.backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self._dim = 1536  # text-embedding-3-small
        else:
            raise ValueError("Unknown EMBEDDING_BACKEND: " + self.backend)

    def _to_2d(self, vecs: List) -> np.ndarray:
        """Coerce a list of vectors (each shape (D,)) into (N, D)."""
        rows = []
        for v in vecs:
            a = np.asarray(v, dtype="float32").reshape(1, -1)
            rows.append(a)
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype="float32")

    @property
    def dim(self) -> int:
        if self.backend == "fastembed":
            if self._dim is None:
                probe = list(self.model.embed(["dim probe", "second probe"]))
                arr = self._to_2d(probe)
                self._dim = int(arr.shape[1])
            return self._dim
        return self._dim

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        if self.backend == "fastembed":
            vecs = list(self.model.embed(texts))  # list of (D,) arrays
            arr = self._to_2d(vecs)               # (N, D)
            if self._dim is None:
                self._dim = int(arr.shape[1])
            return arr
        else:
            out = []
            for t in texts:
                r = self.client.embeddings.create(model=settings.OPENAI_EMBEDDING, input=t)
                out.append(np.array(r.data[0].embedding, dtype="float32").reshape(1, -1))
            return np.vstack(out)  # (N, D)
