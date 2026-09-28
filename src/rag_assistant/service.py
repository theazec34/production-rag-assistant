"""Orchestrate safety checks, retrieval, no-answer policy, and generation."""

from rag_assistant.models import (
    AnswerStatus,
    AskResponse,
    Citation,
    SearchResult,
)
from rag_assistant.providers import GenerationProvider
from rag_assistant.retrieval import LexicalRetriever
from rag_assistant.security import check_question

NO_ANSWER_TEXT = "I do not have enough information in the knowledge base to answer that."
REJECTED_TEXT = "I cannot process this request because it resembles prompt injection."


class RagService:
    def __init__(
        self,
        retriever: LexicalRetriever,
        provider: GenerationProvider,
        *,
        minimum_score: float,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.minimum_score = minimum_score

    async def ask(self, question: str, *, top_k: int = 3) -> AskResponse:
        safety = check_question(question)
        if not safety.safe:
            return AskResponse(
                status=AnswerStatus.REJECTED,
                answer=REJECTED_TEXT,
                confidence=1.0,
                reason=safety.reason,
            )

        results = self.retriever.search(question, top_k=top_k)
        if not results or results[0].score < self.minimum_score:
            return AskResponse(
                status=AnswerStatus.NO_ANSWER,
                answer=NO_ANSWER_TEXT,
                confidence=self._no_answer_confidence(results),
                reason="No retrieved chunk passed the configured relevance threshold.",
            )

        answer = await self.provider.generate(question, results)
        return AskResponse(
            status=AnswerStatus.ANSWERED,
            answer=answer,
            citations=[self._citation(result) for result in results],
            confidence=self._answer_confidence(results),
            reason="Top lexical result passed the configured relevance threshold.",
        )

    def _answer_confidence(self, results: list[SearchResult]) -> float:
        """A visible retrieval heuristic, not a statistical probability."""
        top = results[0].score
        second = results[1].score if len(results) > 1 else 0.0
        threshold_margin = max(0.0, (top - self.minimum_score) / (1 - self.minimum_score))
        rank_margin = max(0.0, top - second)
        return round(min(0.95, 0.5 + 0.3 * threshold_margin + 0.2 * rank_margin), 2)

    def _no_answer_confidence(self, results: list[SearchResult]) -> float:
        top = results[0].score if results else 0.0
        return round(min(0.95, 0.6 + 0.35 * (1 - top)), 2)

    @staticmethod
    def _citation(result: SearchResult) -> Citation:
        return Citation(
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            title=result.chunk.title,
            source=result.chunk.source,
            quote=result.chunk.text[:240],
            score=result.score,
        )
