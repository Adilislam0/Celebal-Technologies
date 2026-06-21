"""
embeddings.py
=============
STEP 3 of the RAG pipeline: Embedding.

WHAT'S HAPPENING CONCEPTUALLY:
An embedding model maps a piece of text to a fixed-length vector (e.g. 384
numbers) such that texts with similar MEANING end up close together in that
vector space, regardless of exact wording. "How do I reset my password?" and
"Steps to recover account access" should land near each other even though
they share almost no words -- this is what makes vector search smarter than
plain keyword matching.

WHY all-MiniLM-L6-v2 specifically:
It's a distilled sentence-transformer model -- small (~80MB), fast on CPU,
384-dimensional output. For a document QA system this size, it gives a strong
quality/speed tradeoff. Larger models (e.g. all-mpnet-base-v2, 768-dim) give
slightly better retrieval accuracy at ~3-4x the compute cost. Swap the
MODEL_NAME constant below if you want to experiment with that tradeoff.

WHY NORMALIZE EMBEDDINGS:
We L2-normalize every vector to unit length. This lets us use a simple dot
product as a stand-in for cosine similarity (cosine similarity of two unit
vectors IS their dot product). FAISS's IndexFlatIP (inner product) index then
becomes a cosine-similarity search "for free" without extra computation per
comparison -- a small trick that meaningfully speeds up search at scale.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        # Loaded once and reused -- loading the model is the expensive part,
        # encoding individual texts afterward is cheap.
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts (used when building the index from chunks).
        Returns an (N, dim) float32 array of unit-normalized vectors.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-length vectors, see docstring above
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a (dim,) float32 vector."""
        return self.embed_texts([query])[0]


if __name__ == "__main__":
    embedder = Embedder()
    vecs = embedder.embed_texts(["A cat sits on a mat.", "Boosting trains models sequentially."])
    print("Shape:", vecs.shape)
    print("Norm of first vector (should be ~1.0):", np.linalg.norm(vecs[0]))
