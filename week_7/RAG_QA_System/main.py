"""
main.py
=======
CLI entry point. Builds the index from data/sample_docs/, then opens an
interactive question-answering loop.

Usage:
    python main.py                          # vector search only, Ollama backend
    python main.py --hybrid                 # + BM25 hybrid search (Step 8)
    python main.py --rerank                 # + cross-encoder reranking (Step 8)
    python main.py --hybrid --rerank         # both optimizations together
    python main.py --backend anthropic       # use Claude API instead of Ollama
                                                (requires ANTHROPIC_API_KEY env var)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import RAGPipeline

DOCS_FOLDER = "data/sample_docs"
INDEX_FOLDER = "data/index"


def main():
    parser = argparse.ArgumentParser(description="Document QA RAG system")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "anthropic"],
                         help="LLM backend to use for generation")
    parser.add_argument("--model", default=None, help="Override the default model name for the chosen backend")
    parser.add_argument("--hybrid", action="store_true", help="Enable BM25 + vector hybrid retrieval")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per query")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the index instead of loading from disk")
    args = parser.parse_args()

    generator_kwargs = {"model": args.model} if args.model else {}
    pipeline = RAGPipeline(generator_backend=args.backend, **generator_kwargs)

    index_exists = Path(INDEX_FOLDER, "index.faiss").exists()
    if index_exists and not args.rebuild:
        print("[main] Loading existing index from disk ...")
        pipeline.load_index(INDEX_FOLDER)
    else:
        pipeline.build_index(DOCS_FOLDER)
        pipeline.save_index(INDEX_FOLDER)

    print("\nDocument QA ready. Type a question, or 'quit' to exit.")
    print(f"Mode: hybrid={args.hybrid}, rerank={args.rerank}, top_k={args.top_k}\n")

    while True:
        query = input("Q> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        answer, chunks = pipeline.answer(
            query, top_k=args.top_k, hybrid=args.hybrid, rerank=args.rerank, return_chunks=True
        )

        print(f"\nA> {answer}\n")
        print("--- Retrieved chunks ---")
        for i, c in enumerate(chunks, start=1):
            preview = c["text"][:80].replace("\n", " ")
            print(f"  [{i}] {c['source']} (chunk {c['chunk_id']}): {preview}...")
        print()


if __name__ == "__main__":
    main()
