# CogniGraph

**Autonomous Multi-Agent RAG System** — query your private document collection using a local LLM-powered agent pipeline.

CogniGraph lets you ask questions about your own documents and get concise, context-aware answers — all running **fully offline** with no external API calls.

## Use Case

CogniGraph is designed for **privacy-preserving research assistance and knowledge-base querying** over a local corpus of documents. Instead of uploading your files to a cloud service, you keep everything on your machine.

**Example scenarios:**
- Ask questions about a collection of research papers or books
- Query internal company documentation without sending data to third parties
- Build a local RAG pipeline for sensitive or proprietary documents
- Experiment with multi-agent LLM workflows using LangGraph and LangChain

## How It Works

```
User Query
  └─► Planner Agent (classifies intent: retrieve / summarize / search)
        └─► Retriever Agent (semantic search in ChromaDB)
              └─► Summarizer Agent (generates answer from retrieved chunks)
                    └─► Response printed to CLI
```

1. **Planner** — Determines what action to take based on the query
2. **Retriever** — Searches ChromaDB for semantically similar document chunks using sentence-transformers embeddings
3. **Summarizer** — Condenses retrieved chunks into a coherent answer using a local Hugging Face model

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| LLM | Hugging Face Transformers (default: `google/flan-t5-small`) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (persistent, local) |
| Interface | CLI |
| Config | YAML + `.env` |
| Containerization | Docker / Docker Compose |

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/#installation)

## Local Setup

```bash
# Clone the repo
git clone <repo-url>
cd cognigraph

# Create .venv and install all dependencies
uv sync

# Configure environment
cp config/env.example .env
# Edit .env if needed (defaults work with flan-t5-small)

# Add your .txt documents to the data/ directory
# (PDF support coming soon)

# Ingest documents into ChromaDB
uv run python data/ingest.py

# Run the CLI
uv run python main.py
```

> No manual activation needed — `uv run` automatically uses the project's `.venv`.

## Docker

```bash
docker-compose up --build
```

Place your `.txt` files in the `data/` directory before starting. Documents are auto-ingested on startup.

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMADB_PATH` | `./chromadb` | Path to ChromaDB persistent storage |
| `MODEL_NAME` | `google/flan-t5-small` | Hugging Face model for text generation |
| `HUGGINGFACE_TOKEN` | — | Required for gated models (e.g., Llama, Mistral) |

### Model Settings (`config/model_settings.yaml`)

```yaml
model:
  name: google/flan-t5-small       # LLM model
  embedding_model: sentence-transformers/all-MiniLM-L6-v2  # Embedding model
  max_length: 512                    # Max generation tokens
  temperature: 0.2                   # Generation temperature
retriever:
  top_k: 5                           # Number of chunks to retrieve
```

## Usage

```bash
uv run python main.py
```

Type a question at the prompt. For example:

```
Enter your query: What is the main idea of document X?
> Action: retrieve
> Summary: The document discusses...

Enter your query: summarize the key points
> Action: summarize
> Summary: ...

Enter your query: exit
```

## Project Structure

```
cognigraph/
├── agents/           # Planner, Retriever, Summarizer agents
├── chromadb/         # ChromaDB persistent vector store (auto-created)
├── config/           # Configuration files
│   ├── env.example
│   └── model_settings.yaml
├── data/             # Place your .txt documents here
│   └── ingest.py     # Document ingestion pipeline
├── eval/             # RAG evaluation suite
│   ├── eval_dataset.py  # Test queries with expected answers
│   ├── metrics.py       # Semantic similarity, F1, faithfulness, coverage
│   └── evaluate.py      # Evaluation harness
├── interface/        # CLI interface
│   └── cli.py
├── memory/           # In-memory conversation history
├── main.py           # Entry point
├── langgraph_flow.py # Agent orchestration workflow
├── pyproject.toml    # Project config & dependencies
├── Dockerfile
└── docker-compose.yml
```

## Evaluation

Quantify RAG pipeline quality with the built-in eval suite:

```bash
uv run python eval/evaluate.py
```

**Metrics tracked:**

| Metric | What it measures |
|--------|-----------------|
| Semantic similarity | Cosine sim between answer and expected answer (via sentence-transformers) |
| Token F1 | Word overlap between answer and expected answer |
| Topic coverage | % of expected topics mentioned in the answer |
| Faithfulness | % of answer sentences supported by retrieved chunks |
| Context relevance | Avg cosine sim between query and retrieved chunks |

Add or edit test queries in `eval/eval_dataset.py`. Results are saved to a timestamped JSON report.

## Roadmap

- [ ] PDF document ingestion (`pdfplumber`/`pypdf` already in dependencies)
- [ ] FastAPI REST API (dependencies already included)
- [ ] LangGraph graph-based orchestration (StateGraph, nodes, edges)
- [ ] Persistent conversation memory (disk-backed)
- [ ] Web UI
