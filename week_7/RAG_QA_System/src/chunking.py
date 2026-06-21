"""
chunking.py
===========
STEP 2 of the RAG pipeline: Chunking.

WHY CHUNKING EXISTS AT ALL:
Embedding models have a limited context window, and more importantly, embedding
an entire long document into ONE vector destroys retrieval precision -- the
vector becomes an average of everything in the doc, so a query about one
specific fact won't match it well. Chunking breaks documents into smaller,
semantically coherent pieces so each piece's embedding represents ONE idea,
which is what makes similarity search actually work.

WHY "CLEAN" CHUNKING MATTERS (the methodology, not just splitting by char count):
Naively slicing text every N characters will cut sentences and even words in
half, e.g. "...neural network achieves hi" | "gh accuracy on...". This produces
embeddings for half-formed thoughts, which retrieves badly. The fix is a
RECURSIVE splitter: try to split on the largest semantic boundary first
(paragraph breaks), and only fall back to a smaller boundary (sentences, then
words) if a chunk is still too big. This keeps chunks coherent.

WHY OVERLAP:
If a sentence describing a concept happens to fall right at a chunk boundary,
splitting it cleanly in two can still separate the setup from the conclusion
of an idea ("Boosting trains models sequentially." | "Each new model corrects
errors of the previous ensemble."). A small overlap (e.g. 15-20% of chunk size)
duplicates the tail of one chunk at the head of the next, so context isn't lost
at the seams -- at the cost of slightly more storage and redundant embeddings.
"""

import re
from typing import List, Dict


# Boundaries to try splitting on, from "biggest semantic unit" to "smallest".
# Order matters: we only fall through to a smaller separator if splitting on
# a bigger one still leaves a chunk that's too long.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on_separator(text: str, separator: str) -> List[str]:
    """Split text on a separator, keeping the separator attached to each piece
    (except the last) so we don't lose sentence-ending punctuation etc."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to all but the final piece
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """
    Core recursive logic: try the first separator; if any resulting piece is
    still bigger than chunk_size, recurse into that piece using the NEXT
    (smaller) separator. This is what makes the splitter "clean" -- it always
    prefers the largest coherent boundary that still fits the size budget.
    """
    if len(text) <= chunk_size or not separators:
        return [text]

    sep, remaining_seps = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)

    result = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            # This piece is still too big -- recurse with a smaller separator
            result.extend(_recursive_split(piece, chunk_size, remaining_seps))
    return result


def _merge_with_overlap(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """
    Greedily pack small pieces (sentences/lines) back together up to
    chunk_size, then carry the overlap (the tail of the previous chunk)
    forward into the next chunk's start.
    """
    chunks = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start the new chunk with overlap from the end of the previous one
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 75,
    separators: List[str] = None,
) -> List[str]:
    """
    Split a single document's text into clean, overlapping chunks.

    Args:
        text: raw document text.
        chunk_size: target max characters per chunk. ~500 chars is a reasonable
            default for sentence-transformer embedding models (roughly 100-125
            tokens) -- big enough to hold a full idea, small enough to stay
            precise on similarity search.
        overlap: characters of overlap carried from the end of one chunk into
            the start of the next. ~15% of chunk_size is a common rule of thumb.
        separators: boundary hierarchy to split on, biggest to smallest.

    Returns:
        List of chunk strings.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    # Step A: recursively split on the biggest boundary that keeps pieces
    # under chunk_size (this gives us small, semantically clean pieces).
    pieces = _recursive_split(text, chunk_size, separators)

    # Step B: pack those small pieces back together up to chunk_size,
    # carrying overlap forward between consecutive chunks.
    chunks = _merge_with_overlap(pieces, chunk_size, overlap)

    return chunks


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 75,
) -> List[Dict[str, str]]:
    """
    Chunk a list of ingested documents (from ingestion.py) into a flat list
    of chunk records, each tagged with its source document and chunk index.

    Returns:
        List of {"text": chunk_str, "source": filename, "chunk_id": int}
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": i,
            })
    return all_chunks


if __name__ == "__main__":
    # Quick manual test
    sample = (
        "Boosting trains models sequentially. Each new model corrects errors "
        "of the previous ensemble.\n\nBagging trains models independently on "
        "bootstrap samples and averages predictions."
    )
    for c in chunk_text(sample, chunk_size=80, overlap=15):
        print(repr(c))
