# CogniGraph — Learning Roadmap

## Phase 1: Graph-Based Orchestration
- [ ] Refactor `langgraph_flow.py` to use LangGraph `StateGraph` with proper nodes, edges, and conditional routing
- [ ] Implement query decomposition agent that breaks complex queries into sub-queries
- [ ] Add retry/fallback logic at the graph level (e.g. if retrieval fails, reformulate and retry)

## Phase 2: Retrieval Engineering
- [ ] Replace naive `\n\n` chunking in `data/ingest.py` with recursive character text splitter
- [ ] Add semantic chunking option (split by embedding similarity)
- [ ] Implement hybrid search: combine ChromaDB semantic search with BM25 keyword search (`rank_bm25`)
- [ ] Add cross-encoder re-ranking step after retrieval (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] Experiment with different `top_k` values and chunk sizes, log impact on eval metrics

## Phase 3: Agentic Reliability
- [ ] Implement Self-RAG / Corrective-RAG reflector node that evaluates retrieval quality and re-formulates queries when needed
- [ ] Add hallucination detection agent using NLI model (`cross-encoder/nli-deberta-v3-base`)
- [ ] Add guardrails for output validation (format, safety, groundedness) using Guardrails AI or custom validators
- [ ] Implement confidence scoring — agent should say "I don't know" when confidence is low

## Phase 4: Production Engineering
- [ ] Add PDF document ingestion using `pdfplumber` / `pypdf` (deps already in pyproject.toml)
- [ ] Add table and image extraction handling for PDFs
- [ ] Replace in-memory `Memory` class with persistent SQLite or ChromaDB-backed store
- [ ] Implement sliding window + summary compression for long conversation histories
- [ ] Add streaming responses to CLI using `TextIteratorStreamer`
- [ ] Build FastAPI REST API (deps already in pyproject.toml) with proper request/response models
- [ ] Add Gradio or Streamlit web UI for chat interface
- [ ] Add authentication and rate limiting to API

## Phase 5: MLOps & Optimization
- [ ] Integrate RAGAS evaluation metrics (answer relevancy, context precision/recall)
- [ ] Add LangSmith or Langfuse tracing for agent observability
- [ ] Log token usage, latency, and cost per query
- [ ] Fine-tune `flan-t5-small` on domain data using LoRA/QLoRA (`peft` library)
- [ ] Benchmark different embedding models and LLMs, track impact on eval scores
- [ ] Add A/B testing framework to compare prompt/model/chunking strategies

## Phase 6: Nice-to-Haves
- [ ] Multi-modal support (handle images and tables in documents)
- [ ] Support for multiple languages
- [ ] Document versioning and incremental ingestion (avoid re-indexing unchanged docs)
- [ ] User feedback loop — collect thumbs up/down, use for prompt/eval iteration
- [ ] Add unit and integration tests for all agents
- [ ] CI/CD pipeline with automated eval on every change
