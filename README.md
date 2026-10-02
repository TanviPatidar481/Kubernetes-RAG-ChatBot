# Kub-Bot — Agentic RAG Assistant

> An advanced Retrieval-Augmented Generation (RAG) application built with FastAPI, LangGraph, Qdrant, Gemini Embeddings, Jina Reranker, NeMo Guardrails, and Portkey.

Kub-Bot is a multi-stage RAG system designed to retrieve relevant information from a domain-specific knowledge base and generate context-grounded responses. The application combines stateful workflow orchestration, conversational memory, authentication, reranking, LLM fallback handling, observability, evaluation, and cloud deployment.

<p align="center">
  <a href="https://kubernetes-rag-chatbot.onrender.com">Live API</a> •
  <a href="https://kub-bot-ui.onrender.com">Web Interface</a> •
  <a href="https://github.com/TanviPatidar481/Kubernetes-RAG-ChatBot">Source Code</a>
</p>

---

## Highlights

- Agentic RAG workflow orchestrated with LangGraph
- Semantic retrieval using Gemini Embeddings and Qdrant
- Two-stage retrieval with deduplication and Jina reranking
- Conversational memory for multi-turn interactions
- LLM routing, fallback handling, and caching through Portkey
- NeMo Guardrails for controlled LLM interactions
- Clerk/JWT authentication for protected API operations
- LangSmith and Logfire for tracing and observability
- RAGAS evaluation across retrieval and generation metrics
- Dockerized deployment with FastAPI backend and Streamlit UI

---

## Architecture

```text
                         +--------------------+
                         |    Streamlit UI    |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         |      FastAPI       |
                         |    Clerk / JWT     |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         |     LangGraph      |
                         |  Stateful Workflow |
                         +---------+----------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
           +-----------------+          +-----------------+
           |    Retrieval    |          |    LLM Layer    |
           +-----------------+          +-----------------+
           | Gemini Embedding|          | Portkey         |
           | Qdrant          |          | LLM Provider    |
           | Deduplication   |          | NeMo Guardrails |
           | Jina Reranker   |          +-----------------+
           +--------+--------+
                    |
                    v
           +-----------------+
           | Context + Prompt|
           +--------+--------+
                    |
                    v
           +-----------------+
           | Final Response  |
           +-----------------+

       Observability: LangSmith + Logfire
       Evaluation:    RAGAS
