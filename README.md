# Production-minded RAG Assistant

An intentionally small reference project for learning the mechanics behind
retrieval-augmented generation (RAG). It runs locally without paid credentials. The default
path uses deterministic BM25-style lexical search and sentence extraction; **neither is an
LLM**. An optional adapter can call an OpenAI-compatible local or hosted generation endpoint.

This is honest portfolio code: it demonstrates engineering decisions and evaluation plumbing,
not a claim that a four-document demo is production-ready.

## Quick start

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run uvicorn rag_assistant.app:app --reload
```

Then open `http://127.0.0.1:8000/docs`, or ask:

```bash
curl -s http://127.0.0.1:8000/v1/ask \
  -H 'content-type: application/json' \
  -d '{"question":"How long can deleted data remain in backups?","top_k":3}'
```

No `.env` is required. Copy `.env.example` to `.env` only to change defaults.

## Request flow

```text
bounded request
    -> prompt-injection heuristics
    -> transparent word-window chunking
    -> in-memory BM25-style lexical ranking
    -> relevance threshold / no-answer decision
    -> extractive sentence OR optional generation provider
    -> structured answer, confidence heuristic, and source citations
```

The application loads the checked-in JSON knowledge base once at startup. Chunks have stable
IDs, document metadata, and overlapping word windows. The retriever tokenizes each chunk,
removes a short stopword list, computes BM25-style scores, and maps them to `[0, 1]` for an
interpretable threshold. A low top score produces a no-answer response instead of guessing.

`confidence` is explicitly a bounded heuristic based on retrieval threshold and rank margins.
It is **not a calibrated probability**. Citations include the source, chunk ID, quote, and
retrieval score so a client can display and audit evidence.

## Generation modes

- `extractive` (default): picks the source sentence with the most lexical overlap. It is
  deterministic, free, and useful for testing the retrieval pipeline. It is not generative AI.
- `openai-compatible`: sends only retrieved context and a grounding instruction to a configured
  `/chat/completions` endpoint. It works with a compatible local server and does not inherently
  require a paid key. See `.env.example`.

Provider output is untrusted text. The adapter bounds output length, but this demo does not
perform semantic citation verification.

## Evaluation

```bash
uv run python -m rag_assistant.evaluation
```

`evaluation/golden_v1.json` is a versioned, reviewable set with answerable, unanswerable, and
adversarial prompts. The command reports:

- retrieval recall@k: fraction of answerable questions where an expected document appears;
- mean reciprocal rank: how early the first expected document appears;
- unanswerable refusal accuracy;
- adversarial rejection accuracy.

These measurements describe only the tiny checked-in set. They are regression signals, not
general ML quality, business impact, latency, or production safety metrics. No answer-quality
metric is invented for the deterministic extractor.

## Quality commands

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
docker build -t production-rag-assistant .
docker run --rm -p 8000:8000 production-rag-assistant
```

CI runs lint, tests with coverage, golden-set evaluation, and a container build. Docker Compose
is intentionally absent because this service has no database, queue, or required companion
process.

## Project map

```text
src/rag_assistant/
  app.py          API and dependency wiring
  chunking.py     overlapping word windows
  retrieval.py    local lexical ranking
  security.py     injection heuristics
  providers.py    extractive and optional HTTP providers
  service.py      retrieval, refusal, citations, confidence
  evaluation.py   golden-set metrics command
data/             non-sensitive sample knowledge base
evaluation/       versioned golden questions
tests/            unit and API behavior tests
```

Modules are separated by responsibility and kept small so a beginner can trace a request.

## Tradeoffs and production gaps

- Lexical retrieval is cheap and explainable but misses synonyms and meaning without shared
  words. A larger system might compare it with embeddings or hybrid retrieval.
- The corpus is loaded into one process and rebuilt at startup. There is no persistent index,
  incremental ingestion, document access control, tenant isolation, or deletion workflow.
- Regex injection detection catches known obvious patterns and can have false positives and
  false negatives. It is defense-in-depth, not a security boundary.
- The threshold was chosen for this sample set. Real deployment requires representative data,
  held-out threshold tuning, and monitoring of score distributions and refusals.
- Generated text is not independently checked for entailment or valid inline citations.
- There is no authentication, authorization, rate limiting, distributed tracing, audit store,
  secrets manager, TLS termination, or high-availability design.
- The sample corpus and golden set are deliberately too small to support broad claims.

## Active-learning disclaimer

There is **no autonomous active-learning loop** in this project. It does not retrain, rewrite
prompts, change thresholds, or add user conversations to the knowledge base. In a real product,
low-confidence questions and explicit feedback could be collected only with consent, privacy
controls, retention limits, access controls, and human review. Approved examples would be
versioned, split to avoid evaluation leakage, tested offline, and deployed through normal code
review. Calling ordinary feedback collection “active learning” would overstate this demo.

## What to be ready to explain

Explain chunk size/overlap, BM25 and lexical limitations, recall@k versus MRR, why no-answer
thresholds need held-out data, why confidence is not probability, how citations preserve
provenance, prompt injection versus ordinary input validation, provider boundaries, async API
behavior, dependency locking, container least privilege, tests versus evaluation, and the
privacy/governance requirements around user feedback.
