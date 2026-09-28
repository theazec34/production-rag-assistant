import asyncio

import pytest
from fastapi.testclient import TestClient

from rag_assistant.app import create_app
from rag_assistant.config import Settings
from rag_assistant.evaluation import evaluate, load_golden_set


def test_health_and_ask_endpoints(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/ask",
            json={"question": "When is support staffed?", "top_k": 2},
        )

    assert health.status_code == 200
    assert health.json()["provider"] == "extractive"
    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert response.json()["citations"][0]["source"].startswith("kb://")


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x"},
        {"question": "valid question", "top_k": 20},
        {"question": "x" * 501},
    ],
)
def test_api_rejects_out_of_bounds_inputs(settings: Settings, payload: dict) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/ask", json=payload)
    assert response.status_code == 422


def test_evaluation_reports_real_counts_and_metrics(settings: Settings) -> None:
    report = asyncio.run(evaluate(settings, top_k=3))

    assert report["counts"] == {
        "answerable": 5,
        "unanswerable": 3,
        "adversarial": 3,
    }
    assert report["retrieval"]["recall_at_3"] == 1.0
    assert report["behavior"]["adversarial_rejection_accuracy"] == 1.0


def test_golden_set_is_versioned(settings: Settings) -> None:
    golden = load_golden_set(settings.evaluation_path)
    assert golden.version == "1.0.0"


def test_invalid_provider_configuration_is_rejected(settings: Settings) -> None:
    invalid = settings.model_copy(update={"provider": "openai-compatible"})
    with pytest.raises(ValueError, match="PROVIDER_BASE_URL"):
        create_app(invalid)
