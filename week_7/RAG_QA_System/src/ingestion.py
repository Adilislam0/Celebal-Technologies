"""
ingestion.py
============
STEP 1 of the RAG pipeline: Document Ingestion.

Job of this module: turn whatever format the source document is in (.txt, .pdf)
into a single, uniform Python representation -- a list of dicts shaped like:

    {"source": "ml_notes.txt", "text": "<full raw text content>"}

Why this matters: every downstream stage (chunking, embedding, retrieval) only
ever needs to deal with plain strings + a source tag. Keeping ingestion isolated
means adding a new file type later (e.g. .docx, .html, a HF dataset) only requires
adding one new loader function here -- nothing else in the pipeline changes.
"""

import os
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def load_text_file(filepath: str) -> str:
    """Read a plain .txt file and return its raw text content."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(filepath: str) -> str:
    """
    Extract text from a PDF file, page by page.

    Note: pypdf does text extraction, not OCR. If the PDF is a scanned image
    (no embedded text layer), this will return an empty/garbled string --
    you'd need an OCR step (e.g. pytesseract) first in that case.
    """
    reader = PdfReader(filepath)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def load_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Walk a folder and load every supported file into a uniform document list.

    Returns:
        List of {"source": filename, "text": raw_text} dicts.
    """
    documents = []
    folder = Path(folder_path)

    for filepath in sorted(folder.iterdir()):
        if not filepath.is_file():
            continue

        suffix = filepath.suffix.lower()
        try:
            if suffix == ".txt":
                text = load_text_file(str(filepath))
            elif suffix == ".pdf":
                text = load_pdf_file(str(filepath))
            else:
                # Skip unsupported file types rather than crashing the whole
                # ingestion run -- one bad file shouldn't block the rest.
                continue

            if text.strip():  # skip empty/unreadable files
                documents.append({"source": filepath.name, "text": text})

        except Exception as e:
            print(f"[ingestion] Warning: failed to load {filepath.name}: {e}")

    return documents


if __name__ == "__main__":
    # Quick manual test: point this at the sample_docs folder
    docs = load_documents("data/sample_docs")
    for doc in docs:
        print(f"Loaded: {doc['source']} ({len(doc['text'])} chars)")
