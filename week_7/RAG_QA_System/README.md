# Document QA RAG System

A modular Retrieval-Augmented Generation pipeline for answering questions
about custom documents (PDFs, text files). Built so each pipeline stage is
isolated and independently testable.

## Architecture

```
data/sample_docs/  --->  ingestion.py  --->  chunking.py  --->  embeddings.py
   (raw docs)          (load to text)      (clean chunks)     (text -> vectors)
                                                                      |
                                                                      v
                                                              vector_store.py
                                                            (FAISS index + metadata)
                                                                      |
        user question  --->  embeddings.py  --->  retriever.py  <----+
                            (embed query)    (vector / hybrid / rerank)
                                                       |
                                                       v
                                                 generator.py
                                          (grounded prompt -> LLM answer)
```

## How this maps to the assignment instructions

| # | Instruction | File |
|---|---|---|
| 1 | Document ingestion (PDF, txt) | `src/ingestion.py` |
| 2 | Clean chunking | `src/chunking.py` |
| 3 | Text -> vector embeddings | `src/embeddings.py` |
| 4 | Vector DB, fast similarity search | `src/vector_store.py` |
| 5 | Query -> query vector | `Retriever._vector_search` in `src/retriever.py` |
| 6 | Retrieval module | `src/retriever.py` |
| 7 | Context + query -> grounded LLM prompt | `src/generator.py` |
| 8 | Optimizations: chunk tuning, hybrid search, reranking | `src/retriever.py` (`hybrid=`, `rerank=` flags), `compare_retrieval.py` |

## Setup

```bash
pip install -r requirements.txt
```

For generation, pick ONE backend:

- **Ollama (free, local, no API key)** — recommended default:
  ```bash
  # install Ollama from https://ollama.com, then:
  ollama pull qwen2.5:0.5b
  ```
- **Anthropic API**:
  ```bash
  export ANTHROPIC_API_KEY="your-key-here"
  ```

## Running it

```bash
cd rag_qa_system

# First run: builds the index from data/sample_docs/ and saves it to data/index/
python main.py

# Subsequent runs reuse the saved index automatically (no re-embedding)
python main.py

# With optimizations enabled
python main.py --hybrid --rerank

# Use Claude instead of local Ollama
python main.py --backend anthropic

# Compare retrieval strategies side-by-side on one query
python compare_retrieval.py "How does boosting differ from bagging?"
```

To use your own documents: drop `.txt` or `.pdf` files into `data/sample_docs/`
and run with `--rebuild` to re-index.

## Key design decisions (and why)

- **Custom recursive chunker instead of a library default** — splits on the
  largest semantic boundary (paragraphs) first, falling back to sentences,
  then words, only when a piece is still too big. Keeps chunks coherent
  instead of cutting mid-sentence. 500-char chunks with 75-char overlap is
  the starting default — tune both based on your documents (denser technical
  text often wants smaller chunks; narrative text can go bigger).
- **all-MiniLM-L6-v2 for embeddings** — 384-dim, fast on CPU, strong
  quality/speed tradeoff for a project this size. Swap for
  `all-mpnet-base-v2` (768-dim) if retrieval quality matters more than speed.
- **FAISS IndexFlatIP with normalized vectors** — exact (not approximate)
  cosine similarity search. Fine up to ~100K+ chunks; would switch to
  IndexHNSWFlat for million-scale corpora where exact search gets slow.
- **Hybrid search (BM25 + vector, fused via Reciprocal Rank Fusion)** — vector
  search alone misses exact keyword/acronym matches; BM25 alone misses
  paraphrases. RRF combines both rankings without needing to hand-tune a
  weighting between two differently-scaled similarity metrics.
- **Two-stage retrieve-then-rerank** — cheap vector/hybrid search pulls a
  generous candidate pool (e.g. top 20), then an expensive cross-encoder
  reranker re-scores just those candidates down to the final top-k. Keeps
  the accurate-but-slow model off the full corpus.
- **Explicit "say you don't know" instruction in the prompt** — the single
  biggest lever against hallucination at the prompting level; without it,
  the model fills gaps with outside knowledge instead of admitting the
  context doesn't answer the question.
