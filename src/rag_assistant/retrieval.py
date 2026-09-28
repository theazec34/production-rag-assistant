"""Dependency-free BM25-style lexical retrieval.

This is a deterministic search component, not an LLM or embedding model.
"""

import math
import re
from collections import Counter

from rag_assistant.models import Chunk, SearchResult

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


class LexicalRetriever:
    """Small in-memory BM25 retriever intended as a transparent local fallback."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks
        self._tokens = [tokenize(chunk.text + " " + chunk.title) for chunk in chunks]
        self._counts = [Counter(tokens) for tokens in self._tokens]
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens)
        document_frequency = Counter(token for tokens in self._tokens for token in set(tokens))
        self._idf = {
            token: math.log(1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        ranked: list[tuple[float, Chunk]] = []
        for chunk, counts, tokens in zip(self.chunks, self._counts, self._tokens, strict=True):
            raw_score = 0.0
            for token in set(query_tokens):
                frequency = counts[token]
                if not frequency:
                    continue
                length_factor = 1 - 0.75 + 0.75 * len(tokens) / self._average_length
                raw_score += (
                    self._idf[token] * (frequency * 2.2) / (frequency + 1.2 * length_factor)
                )
            if raw_score:
                normalized = 1 - math.exp(-raw_score / max(len(set(query_tokens)), 1))
                ranked.append((normalized, chunk))

        ranked.sort(key=lambda item: (-item[0], item[1].id))
        return [SearchResult(chunk=chunk, score=round(score, 4)) for score, chunk in ranked[:top_k]]
