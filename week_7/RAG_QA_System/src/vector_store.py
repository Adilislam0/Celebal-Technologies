"""
vector_store.py
===============
STEP 4 of the RAG pipeline: Vector Database.

WHAT FAISS IS DOING:
FAISS (Facebook AI Similarity Search) is a library for fast nearest-neighbor
search over dense vectors. "Fast similarity matching" at scale is non-trivial:
brute-force comparing a query vector against millions of stored vectors is
O(N) per query. FAISS offers index types that trade exactness for speed:

  - IndexFlatIP: exact search (brute-force inner product, no approximation).
    Fine up to ~100K-1M vectors. This is what we use here -- a beginner-grade
    document QA system will have thousands of chunks at most, so exactness
    is free and there's no reason to give it up.
  - IndexHNSWFlat / IndexIVFFlat: approximate nearest neighbor (ANN) indexes
    that trade a small amount of recall for huge speedups at million+ scale.
    You'd reach for these only once IndexFlatIP search latency becomes a
    bottleneck -- a good thing to mention if asked "how would you scale this?"

WHY WE STORE METADATA SEPARATELY:
FAISS only stores vectors and returns integer IDs -- it doesn't know anything
about "source file" or "chunk text". So we keep a parallel Python list
(self.metadata) where metadata[i] describes the vector at FAISS index
position i. This is the standard pattern for any raw vector index.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        """
        Args:
            dim: dimensionality of the embedding vectors (e.g. 384 for
                 all-MiniLM-L6-v2). Must match the embedder's output size.
        """
        self.dim = dim
        # Inner product index -- equivalent to cosine similarity since our
        # embeddings are unit-normalized (see embeddings.py docstring).
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []  # metadata[i] <-> vector at position i

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        """
        Add a batch of (embedding, metadata) pairs to the store.

        Args:
            embeddings: (N, dim) float32 array.
            metadatas: list of N dicts, e.g. {"text": ..., "source": ..., "chunk_id": ...}
        """
        assert embeddings.shape[0] == len(metadatas), "embeddings/metadata count mismatch"
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find the top_k most similar stored vectors to the query vector.

        Returns:
            List of (metadata_dict, similarity_score) tuples, sorted best-first.
        """
        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS returns -1 when fewer than top_k vectors exist
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, folder_path: str) -> None:
        """Persist the FAISS index + metadata to disk so we don't re-embed every run."""
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        with open(folder / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(folder / "config.json", "w") as f:
            json.dump({"dim": self.dim}, f)

    @classmethod
    def load(cls, folder_path: str) -> "VectorStore":
        """Load a previously saved index + metadata back from disk."""
        folder = Path(folder_path)
        with open(folder / "config.json") as f:
            config = json.load(f)

        store = cls(dim=config["dim"])
        store.index = faiss.read_index(str(folder / "index.faiss"))
        with open(folder / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        return store

    def __len__(self) -> int:
        return self.index.ntotal


if __name__ == "__main__":
    # Quick manual test with random vectors (no embedding model needed)
    dim = 8
    store = VectorStore(dim=dim)
    fake_vecs = np.random.rand(3, dim).astype("float32")
    fake_vecs /= np.linalg.norm(fake_vecs, axis=1, keepdims=True)  # normalize
    fake_meta = [{"text": f"chunk {i}", "source": "test.txt", "chunk_id": i} for i in range(3)]

    store.add(fake_vecs, fake_meta)
    print("Stored vectors:", len(store))

    results = store.search(fake_vecs[0], top_k=2)
    for meta, score in results:
        print(meta, "score:", round(score, 4))
