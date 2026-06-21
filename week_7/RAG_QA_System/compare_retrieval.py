"""
compare_retrieval.py
=====================
Step 8 deliverable: side-by-side comparison of retrieval strategies on the
same query, so you can SEE the effect of each optimization rather than just
taking it on faith. Run this after building the index once via main.py.

Usage:
    python compare_retrieval.py "your question here"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import Embedder
from vector_store import VectorStore
from retriever import Retriever

INDEX_FOLDER = "data/index"


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does boosting differ from bagging?"

    embedder = Embedder()
    store = VectorStore.load(INDEX_FOLDER)
    retriever = Retriever(store, embedder)

    print(f"Query: {query}\n")

    configs = [
        ("Vector only", dict(hybrid=False, rerank=False)),
        ("Hybrid (vector + BM25)", dict(hybrid=True, rerank=False)),
        ("Vector + Rerank", dict(hybrid=False, rerank=True)),
        ("Hybrid + Rerank", dict(hybrid=True, rerank=True)),
    ]

    for label, kwargs in configs:
        print(f"--- {label} ---")
        results = retriever.retrieve(query, top_k=3, **kwargs)
        for i, c in enumerate(results, start=1):
            preview = c["text"][:90].replace("\n", " ")
            print(f"  [{i}] {c['source']} (chunk {c['chunk_id']}): {preview}...")
        print()


if __name__ == "__main__":
    main()
