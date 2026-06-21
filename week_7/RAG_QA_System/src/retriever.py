"""
retriever.py
============
STEPS 5, 6, and part of 8 of the RAG pipeline: Query Encoding, Retrieval,
and Optimization (hybrid search + reranking).

STEP 5 -- Query embedding:
The incoming question gets embedded with the SAME model used for the chunks.
This is non-negotiable: if you embed chunks with model A and queries with
model B, their vectors live in unrelated spaces and similarity scores are
meaningless. (Easy bug to introduce if you "upgrade" only one side later.)

STEP 6 -- Vector retrieval:
Plain nearest-neighbor search in embedding space. Good at matching MEANING
("how do I undo a commit" matches "reverting changes" even with zero shared
words) but can miss exact terms -- e.g. it may not strongly favor a chunk
just because it contains the literal acronym "XGBoost" if the chunk's overall
semantic vector is pulled elsewhere by surrounding text.

STEP 8 -- Hybrid search (the fix for vector search's blind spot):
BM25 is a classic keyword-based ranking algorithm (a refinement of TF-IDF)
that scores documents by term overlap with the query, weighted by how rare
that term is across the corpus. It's exactly the opposite failure profile of
vector search: great at exact terms, blind to paraphrasing. Combining both
via Reciprocal Rank Fusion (RRF) gives you both strengths -- this is the
single highest-leverage "advanced RAG" trick for general-purpose document QA.

RRF formula: for each chunk, score = sum over each ranking method of
1 / (k + rank_in_that_method), where rank is 1-indexed and k is a constant
(60 is a common default) that discounts the importance of being ranked #1
vs #2 vs #3. A chunk ranked highly by BOTH methods wins; reliance on a single
method gets a smaller score than agreement.

STEP 8 -- Reranking (the fix for "top-k by similarity isn't always top-k by relevance"):
Embedding-based retrieval scores query and chunk independently, then compares
vectors -- fast, but loses information (it never lets the query and chunk text
"interact" with each other word-by-word). A cross-encoder reranker instead
feeds the (query, chunk) pair TOGETHER through a transformer, letting it
attend across both texts simultaneously -- much more accurate, but far more
compute-expensive per pair. The standard pattern: use cheap vector search to
get a generous top-20 candidates, then use the expensive reranker to re-score
and keep only the true top-5. This two-stage "retrieve-then-rerank" pattern
is the most common production RAG architecture for exactly this cost/accuracy
reason.
"""

from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from embeddings import Embedder
from vector_store import VectorStore


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25. A production system
    might add stemming/stopword removal, but this is enough for a clean demo."""
    return text.lower().split()


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

        # Build the BM25 index once over all stored chunk texts. This is what
        # enables hybrid search -- a separate keyword index alongside the
        # vector index, both pointing at the same underlying chunks.
        self.chunk_texts = [m["text"] for m in vector_store.metadata]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.chunk_texts])

        self._reranker = None  # lazy-loaded only if rerank=True is used

    def _get_reranker(self):
        """Lazily load the cross-encoder reranker only when first needed --
        it's a separate model download, no point loading it if unused."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            # ms-marco-MiniLM-L-6-v2: small, fast cross-encoder fine-tuned on
            # query-passage relevance (MS MARCO ranking dataset) -- a standard
            # off-the-shelf choice for reranking.
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        query_vec = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k)
        return [meta for meta, score in results]

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.vector_store.metadata[i] for i in top_indices]

    def _hybrid_search(self, query: str, top_k: int, candidate_pool: int = 20) -> List[Dict]:
        """Reciprocal Rank Fusion of vector search + BM25 search rankings."""
        vec_results = self._vector_search(query, top_k=candidate_pool)
        bm25_results = self._bm25_search(query, top_k=candidate_pool)

        # Use chunk text as a stable key to merge rankings from both lists
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict] = {}
        k = 60  # RRF discount constant, standard default

        for rank, chunk in enumerate(vec_results, start=1):
            key = chunk["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            chunk_lookup[key] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            key = chunk["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            chunk_lookup[key] = chunk

        ranked_keys = sorted(rrf_scores.keys(), key=lambda k_: rrf_scores[k_], reverse=True)
        return [chunk_lookup[k_] for k_ in ranked_keys[:top_k]]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        rerank: bool = False,
        rerank_candidate_pool: int = 20,
    ) -> List[Dict]:
        """
        Main entry point: retrieve the top_k most relevant chunks for a query.

        Args:
            query: the user's question.
            top_k: number of chunks to return for the generator.
            hybrid: if True, combine vector + BM25 search via RRF (Step 8 optimization).
            rerank: if True, apply cross-encoder reranking on top of the initial
                     candidate pool (Step 8 optimization). Can be combined with hybrid.
            rerank_candidate_pool: how many candidates to pull before reranking
                     down to top_k. Wider pool = reranker has more to work with,
                     at the cost of more cross-encoder calls.

        Returns:
            List of chunk metadata dicts, best-first.
        """
        if rerank:
            pool_size = rerank_candidate_pool
            candidates = (
                self._hybrid_search(query, top_k=pool_size)
                if hybrid
                else self._vector_search(query, top_k=pool_size)
            )
            if not candidates:
                return []

            reranker = self._get_reranker()
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.predict(pairs)

            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in ranked[:top_k]]

        if hybrid:
            return self._hybrid_search(query, top_k=top_k)

        return self._vector_search(query, top_k=top_k)
