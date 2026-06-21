"""
pipeline.py
===========
Ties all six modules into one cohesive RAG system with two entry points:

    pipeline.build_index(folder)   -> runs Steps 1-4 (ingest, chunk, embed, store)
    pipeline.answer(query)         -> runs Steps 5-7 (query embed, retrieve, generate)

Splitting these into two phases matters in practice: building the index is
slow (one-time cost, scales with corpus size) while answering a query should
be fast (runs on every user interaction). You build the index once, save it
to disk, and load it back on every subsequent run instead of re-embedding
your entire document set on every query.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingestion import load_documents
from chunking import chunk_documents
from embeddings import Embedder
from vector_store import VectorStore
from retriever import Retriever
from generator import get_generator, build_prompt


class RAGPipeline:
    def __init__(self, generator_backend: str = "ollama", **generator_kwargs):
        self.embedder = Embedder()
        self.generator = get_generator(backend=generator_backend, **generator_kwargs)
        self.vector_store = None
        self.retriever = None

    def build_index(self, docs_folder: str, chunk_size: int = 500, overlap: int = 75) -> None:
        """Steps 1-4: ingest documents, chunk them, embed the chunks, store the vectors."""
        print(f"[pipeline] Loading documents from {docs_folder} ...")
        documents = load_documents(docs_folder)
        print(f"[pipeline] Loaded {len(documents)} document(s).")

        print(f"[pipeline] Chunking (chunk_size={chunk_size}, overlap={overlap}) ...")
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        print(f"[pipeline] Produced {len(chunks)} chunks.")

        print("[pipeline] Embedding chunks ...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        print("[pipeline] Building vector store ...")
        self.vector_store = VectorStore(dim=embeddings.shape[1])
        self.vector_store.add(embeddings, chunks)

        self.retriever = Retriever(self.vector_store, self.embedder)
        print(f"[pipeline] Index ready -- {len(self.vector_store)} chunks indexed.")

    def save_index(self, folder_path: str) -> None:
        self.vector_store.save(folder_path)

    def load_index(self, folder_path: str) -> None:
        self.vector_store = VectorStore.load(folder_path)
        self.retriever = Retriever(self.vector_store, self.embedder)

    def answer(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        rerank: bool = False,
        return_chunks: bool = False,
    ):
        """Steps 5-7: embed the query, retrieve relevant chunks, generate a grounded answer."""
        if self.retriever is None:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")

        chunks = self.retriever.retrieve(query, top_k=top_k, hybrid=hybrid, rerank=rerank)
        prompt = build_prompt(query, chunks)
        answer_text = self.generator.generate(prompt)

        if return_chunks:
            return answer_text, chunks
        return answer_text
