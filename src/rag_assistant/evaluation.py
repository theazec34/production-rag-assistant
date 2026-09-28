"""Evaluate retrieval and refusal behavior against the versioned golden set."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

from rag_assistant.app import build_service
from rag_assistant.config import Settings
from rag_assistant.models import AnswerStatus


class GoldenCase(BaseModel):
    id: str
    category: Literal["answerable", "unanswerable", "adversarial"]
    question: str
    expected_document_ids: list[str] = Field(default_factory=list)


class GoldenSet(BaseModel):
    version: str
    cases: list[GoldenCase]


def load_golden_set(path: Path) -> GoldenSet:
    return TypeAdapter(GoldenSet).validate_json(path.read_text(encoding="utf-8"))


async def evaluate(settings: Settings, *, top_k: int = 3) -> dict[str, object]:
    golden = load_golden_set(settings.evaluation_path)
    service = build_service(settings)
    answerable = [case for case in golden.cases if case.category == "answerable"]
    unanswerable = [case for case in golden.cases if case.category == "unanswerable"]
    adversarial = [case for case in golden.cases if case.category == "adversarial"]

    retrieval_hits = 0
    reciprocal_ranks: list[float] = []
    for case in answerable:
        results = service.retriever.search(case.question, top_k=top_k)
        ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if result.chunk.document_id in case.expected_document_ids
        ]
        if ranks:
            retrieval_hits += 1
            reciprocal_ranks.append(1 / ranks[0])
        else:
            reciprocal_ranks.append(0.0)

    unanswerable_correct = 0
    for case in unanswerable:
        response = await service.ask(case.question, top_k=top_k)
        unanswerable_correct += response.status == AnswerStatus.NO_ANSWER

    adversarial_correct = 0
    for case in adversarial:
        response = await service.ask(case.question, top_k=top_k)
        adversarial_correct += response.status == AnswerStatus.REJECTED

    return {
        "golden_set_version": golden.version,
        "top_k": top_k,
        "counts": {
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "adversarial": len(adversarial),
        },
        "retrieval": {
            f"recall_at_{top_k}": round(retrieval_hits / len(answerable), 3),
            "mean_reciprocal_rank": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
        },
        "behavior": {
            "unanswerable_refusal_accuracy": round(unanswerable_correct / len(unanswerable), 3),
            "adversarial_rejection_accuracy": round(adversarial_correct / len(adversarial), 3),
        },
        "note": (
            "Metrics describe this small checked-in set only; they are not model-quality "
            "or production-performance claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path)
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    settings = Settings(
        evaluation_path=args.golden_set
        if args.golden_set is not None
        else Settings().evaluation_path
    )
    print(json.dumps(asyncio.run(evaluate(settings, top_k=args.top_k)), indent=2))


if __name__ == "__main__":
    main()
