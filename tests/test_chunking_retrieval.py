import pytest

from rag_assistant.chunking import chunk_document
from rag_assistant.models import Document
from rag_assistant.retrieval import LexicalRetriever, tokenize


def test_chunking_has_stable_overlap() -> None:
    document = Document(
        id="guide",
        title="Guide",
        source="test://guide",
        text="one two three four five six seven",
    )
    chunks = chunk_document(document, size_words=4, overlap_words=1)

    assert [chunk.id for chunk in chunks] == ["guide#chunk-0", "guide#chunk-1"]
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "four five six seven"


@pytest.mark.parametrize("size,overlap", [(0, 0), (3, -1), (3, 3)])
def test_chunking_rejects_invalid_configuration(size: int, overlap: int) -> None:
    document = Document(id="x", title="x", source="x", text="some text")
    with pytest.raises(ValueError):
        chunk_document(document, size_words=size, overlap_words=overlap)


def test_empty_document_produces_no_chunks() -> None:
    document = Document(id="x", title="x", source="x", text="  ")
    assert chunk_document(document) == []


def test_retriever_ranks_relevant_document(service) -> None:
    results = service.retriever.search("minimum password length", top_k=2)
    assert results[0].chunk.document_id == "security"
    assert 0 < results[0].score <= 1


def test_retriever_handles_empty_query(service) -> None:
    assert service.retriever.search("the and is") == []
    assert "password" in tokenize("The PASSWORD is long")


def test_retriever_requires_chunks() -> None:
    with pytest.raises(ValueError, match="At least one"):
        LexicalRetriever([])
