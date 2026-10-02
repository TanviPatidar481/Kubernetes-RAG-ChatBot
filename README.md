# Kub-Bot --- Agentic RAG Assistant

> An advanced Retrieval-Augmented Generation (RAG) application built
> with FastAPI, LangGraph, Qdrant, Gemini Embeddings, Jina Reranker,
> NVIDIA NeMo Guardrails, and Portkey.

Kub-Bot is a multi-stage RAG system designed to retrieve relevant
information from a domain-specific knowledge base and generate
context-grounded responses. The application combines stateful workflow
orchestration, conversational memory, authentication, reranking, LLM
fallback handling, observability, evaluation, and cloud deployment.

[Live API](https://kubernetes-rag-chatbot.onrender.com/) · [Web
Interface](https://kub-bot-ui.onrender.com/) · [Source
Code](https://github.com/TanviPatidar481/Kubernetes-RAG-ChatBot)

------------------------------------------------------------------------

## Highlights

-   Agentic RAG workflow orchestrated with LangGraph.
-   Semantic retrieval using Gemini Embeddings and Qdrant.
-   Two-stage retrieval with deduplication and Jina reranking.
-   Conversational memory for multi-turn interactions.
-   LLM routing, fallback handling, and caching through Portkey.
-   NVIDIA NeMo Guardrails for controlled LLM interactions.
-   Clerk/JWT authentication for protected API operations.
-   LangSmith and Logfire for tracing and observability.
-   RAGAS evaluation across retrieval and generation metrics.
-   Dockerized deployment with a FastAPI backend and Streamlit UI.

------------------------------------------------------------------------

## Architecture

``` text
                         +--------------------+
                         |     Streamlit UI   |
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
```

### RAG Pipeline

Kub-Bot extends the traditional RAG flow:

``` text
Query → Embedding → Vector Search → Context → LLM → Answer
```

into a multi-stage workflow:

``` text
User Query
    |
    v
Authentication
    |
    v
Conversation Context
    |
    v
Gemini Embedding
    |
    v
Qdrant Vector Search
    |
    v
Candidate Documents
    |
    v
Deduplication
    |
    v
Jina Reranking
    |
    v
Relevant Context
    |
    v
LangGraph Generation Workflow
    |
    v
LLM
    |
    v
NeMo Guardrails
    |
    v
Final Response
```

This separates candidate retrieval from fine-grained relevance ranking
before generation.

------------------------------------------------------------------------

## Retrieval

### Gemini Embeddings

User queries are converted into vector representations using:

  Setting       Value
  ------------- ------------------------------
  Model         `gemini-embedding-2-preview`
  Vector size   `3072`

### Qdrant

The generated embeddings are searched against the Qdrant collection:

``` text
enterprise_rag_v2
```

### Jina Reranker

Retrieved candidates are deduplicated and then reranked using:

``` text
Jina Reranker v2 — Base Multilingual
```

This provides a second-stage relevance refinement after vector
similarity search.

------------------------------------------------------------------------

## Conversational Memory

Kub-Bot maintains conversational context to support multi-turn
interactions.

**Example:**

``` text
User:      What is Kubernetes?

Assistant: Kubernetes is ...

User:      What are its main components?

Assistant: The main components are ...
```

The second query can be interpreted using the preceding conversation
rather than being treated as an isolated request.

------------------------------------------------------------------------

## Guardrails and Reliability

### NVIDIA NeMo Guardrails

NVIDIA NeMo Guardrails is integrated as a control layer for LLM
interactions.

### Portkey

Portkey provides infrastructure for:

-   LLM routing.
-   Fallback handling.
-   Caching.

These components help manage external model dependencies and provide
additional reliability mechanisms around LLM requests.

------------------------------------------------------------------------

## Authentication

The FastAPI backend uses Clerk/JWT authentication for protected
operations.

``` text
Client
  |
  v
Authentication
  |
  v
JWT Validation
  |
  v
FastAPI
  |
  v
RAG Workflow
  |
  v
Response
```

------------------------------------------------------------------------

## Observability

Kub-Bot uses two complementary observability systems:

  Tool        Purpose
  ----------- ---------------------------------------
  LangSmith   LangChain/LangGraph tracing
  Logfire     Application and FastAPI observability

This provides visibility into both application execution and AI workflow
behaviour.

------------------------------------------------------------------------

## Evaluation

Kub-Bot includes a separate RAGAS evaluation workflow covering retrieval
and generation quality.

  Metric                 Score
  -------------------- -------
  Faithfulness            0.98
  Answer Relevancy        0.59
  Context Precision       0.72
  Context Recall          0.83
  Answer Correctness      0.51

> **Evaluation note:** These values are from a specific local evaluation
> run and should not be interpreted as permanent system-wide accuracy.

The evaluation helps distinguish issues related to retrieval, context
selection, grounding, relevance, and answer generation.

------------------------------------------------------------------------

## Technology Stack

  Category             Technologies
  -------------------- ----------------------------
  Language             Python 3.11.9
  Backend              FastAPI, Uvicorn, Pydantic
  Workflow             LangChain, LangGraph
  Embeddings           Gemini Embeddings
  Vector Database      Qdrant
  Reranking            Jina Reranker
  LLM Infrastructure   Portkey
  Guardrails           NVIDIA NeMo Guardrails
  Authentication       Clerk / JWT
  Frontend             Streamlit
  Observability        LangSmith, Logfire
  Evaluation           RAGAS
  Deployment           Docker, Render

------------------------------------------------------------------------

## Project Structure

``` text
Kubernetes-RAG-ChatBot/
│
├── app/                  # FastAPI backend and RAG application
├── evals/                # RAG evaluation workflow
├── DATA/                 # Knowledge-base data
├── ui/                   # Streamlit interface
│   ├── app.py
│   ├── Logo.png
│   └── bg.png
│
├── .streamlit/           # Streamlit configuration
├── Dockerfile
├── requirements.txt
├── .python-version
└── .gitignore
```

*The structure above highlights the main project directories and files;
the actual repository may contain additional modules and configuration
files.*

------------------------------------------------------------------------

## Getting Started

### Prerequisites

-   Python 3.11.
-   Git.
-   A Qdrant instance.
-   Required LLM and embedding credentials.
-   Jina API credentials.
-   Clerk configuration.
-   Portkey configuration.
-   LangSmith and/or Logfire credentials if observability is enabled.

### 1. Clone the Repository

``` bash
git clone https://github.com/TanviPatidar481/Kubernetes-RAG-ChatBot.git
cd Kubernetes-RAG-ChatBot
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Configure the credentials required by the application for the relevant
services:

-   LLM and embedding provider.
-   Qdrant.
-   Jina.
-   Portkey.
-   Clerk.
-   LangSmith.
-   Logfire.

Use the environment variable names expected by the application
configuration.

> **Security:** Never commit API keys, JWT secrets, database
> credentials, or other sensitive values to the repository. Store
> secrets in environment variables or an appropriate secrets manager.

### 5. Start the FastAPI Backend

``` bash
uvicorn app.main:app --reload
```

API documentation:

<http://127.0.0.1:8000/docs>

### 6. Start the Streamlit Interface

``` bash
streamlit run ui/app.py
```

------------------------------------------------------------------------

## Engineering Challenges

### Reranking Memory Usage

A local Sentence Transformer reranking approach introduced significant
memory requirements during development.

FlashRank was investigated as a lightweight alternative, while the
deployed application uses the hosted Jina Reranker API.

### NeMo Guardrails Memory Overhead

During local testing, adding NeMo Guardrails increased observed memory
usage approximately from:

``` text
~301 MB → ~574 MB
```

with an HTTP peak of approximately:

``` text
~595 MB
```

This highlighted the resource considerations involved in deploying
multi-component AI applications.

### Observability Authentication

During development, the `/query` API could successfully return a
response while LangSmith tracing produced a `401` authentication error.

This demonstrated that application execution and observability
integrations can fail independently.

------------------------------------------------------------------------

## Future Improvements

-   Hybrid retrieval combining dense and sparse search.
-   Improved query routing and query transformation.
-   Better handling of long conversations and conversation
    summarization.
-   Streaming responses.
-   Expanded automated evaluation and regression testing.
-   Deployment resource optimization.
-   Additional failure-recovery and retry strategies.

------------------------------------------------------------------------

## Live Deployment

  -----------------------------------------------------------------------------------
  Component                           URL
  ----------------------------------- -----------------------------------------------
  Backend API                         <https://kubernetes-rag-chatbot.onrender.com>

  Web Interface                       <https://kub-bot-ui.onrender.com>
  -----------------------------------------------------------------------------------

Deployment availability depends on the current hosting environment and
configured external services.

------------------------------------------------------------------------

## Author

**Tanvi Patidar**\
B.Tech --- Computer Science & Engineering\
Medi-Caps University, Indore

-   **GitHub:** [TanviPatidar481](https://github.com/TanviPatidar481)
-   **Project Repository:**
    [Kubernetes-RAG-ChatBot](https://github.com/TanviPatidar481/Kubernetes-RAG-ChatBot)

------------------------------------------------------------------------

## Project Summary

Kub-Bot demonstrates the engineering of a multi-component RAG
application combining:

``` text
FastAPI
   +
LangGraph
   +
Qdrant
   +
Gemini Embeddings
   +
Jina Reranker
   +
Portkey
   +
NeMo Guardrails
   +
Clerk / JWT
   +
LangSmith / Logfire
   +
RAGAS
   +
Streamlit
```

The project focuses on retrieval quality, reranking, stateful workflow
orchestration, conversational context, controlled LLM interactions,
authentication, observability, evaluation, and deployment.

By combining these components into a single application, Kub-Bot
provides an implementation of an end-to-end, domain-specific Agentic RAG
assistant.
