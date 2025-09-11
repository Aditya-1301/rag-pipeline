import os, json
import numpy as np
import faiss
from .config import settings

class DocStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()

    def add(self, docs):
        with open(self.path, "a", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def all(self):
        out = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

class FaissIndex:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            self.index = faiss.read_index(path)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vecs: np.ndarray):
        faiss.normalize_L2(vecs)
        self.index.add(vecs)

    def search(self, vecs: np.ndarray, top_k: int):
        faiss.normalize_L2(vecs)
        scores, idxs = self.index.search(vecs, top_k)
        return scores, idxs

    def save(self):
        faiss.write_index(self.index, self.path)

def reset_storage():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    for p in [settings.INDEX_PATH, settings.DOCSTORE_PATH]:
        if os.path.exists(p):
            os.remove(p)

docstore = DocStore(settings.DOCSTORE_PATH)
