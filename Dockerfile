FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.12/site-packages" \
    RAG_KNOWLEDGE_PATH="/app/data/knowledge_base.json" \
    RAG_EVALUATION_PATH="/app/evaluation/golden_v1.json"

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /install /install
COPY data ./data
COPY evaluation ./evaluation
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "rag_assistant.app:app", "--host", "0.0.0.0", "--port", "8000"]
