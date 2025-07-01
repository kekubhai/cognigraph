# CogniGraph

Autonomous Multi-Agent RAG System using LangGraph, LangChain, ChromaDB, and Hugging Face Transformers.

## Features
- Multi-agent workflow: Planner, Retriever, Summarizer
- ChromaDB vector storage
- Hugging Face LLMs (Mistral/BGE)
- CLI interface
- Dockerized for local/cloud deployment

## Quickstart
```bash
docker-compose up --build
```

## Configuration
- Copy `config/env.example` to `.env` and set your variables.
- Edit `config/model_settings.yaml` for model/retriever settings.
