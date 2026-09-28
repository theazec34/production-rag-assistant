"""Load the deliberately small, version-controlled knowledge base."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from rag_assistant.models import Document

_DOCUMENTS = TypeAdapter(list[Document])


def load_documents(path: Path) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _DOCUMENTS.validate_python(raw["documents"])
