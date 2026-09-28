from pathlib import Path

import pytest

from rag_assistant.app import build_service
from rag_assistant.config import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        knowledge_path=ROOT / "data" / "knowledge_base.json",
        evaluation_path=ROOT / "evaluation" / "golden_v1.json",
    )


@pytest.fixture
def service(settings: Settings):
    return build_service(settings)
