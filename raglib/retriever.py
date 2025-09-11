from typing import List, Dict, Any
import numpy as np
from .embed import Embedder
from .store import FaissIndex, docstore
from .config import settings

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.index = FaissIndex(self.embedder.dim, settings.INDEX_PATH)

    def add(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        vecs = self.embedder.batch_embed(texts)
        self.index.add(vecs)
        self.index.save()
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = top_k or settings.TOP_K
        qv = self.embedder.batch_embed([query])
        scores, idxs = self.index.search(qv, k)
        alld = docstore.all()
        hits = []
        for i, score in zip(idxs[0], scores[0]):
            if 0 <= i < len(alld):
                d = alld[i].copy()
                d["score"] = float(score)
                hits.append(d)
        return hits
