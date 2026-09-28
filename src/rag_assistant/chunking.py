"""Transparent word-window chunking suitable for small text collections."""

import re

from rag_assistant.models import Chunk, Document

_WORD = re.compile(r"\S+")


def chunk_document(
    document: Document,
    *,
    size_words: int = 100,
    overlap_words: int = 20,
) -> list[Chunk]:
    """Split a document into overlapping word windows with stable identifiers."""
    if size_words <= 0:
        raise ValueError("size_words must be positive")
    if overlap_words < 0 or overlap_words >= size_words:
        raise ValueError("overlap_words must be between 0 and size_words - 1")

    words = _WORD.findall(document.text)
    if not words:
        return []

    chunks: list[Chunk] = []
    step = size_words - overlap_words
    for position, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size_words]
        if not window:
            break
        chunks.append(
            Chunk(
                id=f"{document.id}#chunk-{position}",
                document_id=document.id,
                title=document.title,
                source=document.source,
                text=" ".join(window),
                position=position,
            )
        )
        if start + size_words >= len(words):
            break
    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    size_words: int = 100,
    overlap_words: int = 20,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            size_words=size_words,
            overlap_words=overlap_words,
        )
    ]
