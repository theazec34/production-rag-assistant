"""FastAPI application and dependency wiring."""

from fastapi import FastAPI, HTTPException

from rag_assistant.chunking import chunk_documents
from rag_assistant.config import Settings, get_settings
from rag_assistant.knowledge import load_documents
from rag_assistant.models import AskRequest, AskResponse
from rag_assistant.providers import ExtractiveProvider, OpenAICompatibleProvider
from rag_assistant.retrieval import LexicalRetriever
from rag_assistant.service import RagService


def build_service(settings: Settings) -> RagService:
    documents = load_documents(settings.knowledge_path)
    chunks = chunk_documents(
        documents,
        size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    retriever = LexicalRetriever(chunks)
    if settings.provider == "extractive":
        provider = ExtractiveProvider()
    elif settings.provider == "openai-compatible" and settings.provider_base_url:
        provider = OpenAICompatibleProvider(
            base_url=settings.provider_base_url,
            model=settings.provider_model,
            api_key=settings.provider_api_key,
            timeout_seconds=settings.provider_timeout_seconds,
        )
    else:
        raise ValueError(
            "RAG_PROVIDER must be 'extractive', or 'openai-compatible' with "
            "RAG_PROVIDER_BASE_URL set"
        )
    return RagService(
        retriever,
        provider,
        minimum_score=settings.minimum_retrieval_score,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    current_settings = settings or get_settings()
    service = build_service(current_settings)
    application = FastAPI(
        title="Production-minded RAG Assistant",
        version="0.1.0",
        description="Educational local lexical retrieval with explicit citations.",
    )

    @application.get("/health")
    async def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "provider": current_settings.provider,
            "chunks": len(service.retriever.chunks),
        }

    @application.post("/v1/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        try:
            return await service.ask(request.question, top_k=request.top_k)
        except Exception as exc:
            if current_settings.provider == "openai-compatible":
                raise HTTPException(
                    status_code=502,
                    detail="The configured generation provider is unavailable.",
                ) from exc
            raise

    return application


app = create_app()
