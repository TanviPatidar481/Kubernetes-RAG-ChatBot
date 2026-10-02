# Kub-Bot --- Agentic RAG Assistant

A domain-specific Retrieval-Augmented Generation (RAG) application built
with **FastAPI, LangGraph, Qdrant, Gemini Embeddings, Jina Reranker,
NeMo Guardrails, and Portkey**.

Kub-Bot retrieves and reranks relevant knowledge-base content before
generating context-grounded answers. It combines conversational memory,
authentication, LLM reliability controls, observability, and RAG
evaluation in one application.

**[Live API](https://kubernetes-rag-chatbot.onrender.com/)** · **[Web
Interface](https://kub-bot-ui.onrender.com/)** · **[GitHub
Repository](https://github.com/TanviPatidar481/Kubernetes-RAG-ChatBot)**

------------------------------------------------------------------------

## Features

-   **Agentic workflow:** Stateful RAG orchestration with LangGraph.
-   **Semantic retrieval:** Gemini embeddings with Qdrant vector search.
-   **Reranking:** Deduplication and Jina reranking to refine retrieved
    results.
-   **Conversational memory:** Context-aware multi-turn interactions.
-   **LLM reliability:** Portkey routing, fallback handling, and
    caching.
-   **Guardrails:** NVIDIA NeMo Guardrails for controlled model
    interactions.
-   **Authentication:** Clerk/JWT-protected API operations.
-   **Observability:** LangSmith tracing and Logfire application
    monitoring.
-   **Evaluation:** RAGAS metrics for retrieval and generation quality.

## Architecture

``` text
                 User / Streamlit UI
                         |
                         v
                    FastAPI API
                  (Clerk / JWT)
                         |
                         v
                   LangGraph Flow
                         |
              +----------+----------+
              |                     |
              v                     v
         Retrieval               LLM Layer
              |                     |
        Gemini Embeddings         Portkey
              |                     |
           Qdrant              LLM Provider
              |                     |
        Deduplication          NeMo Guardrails
              |
        Jina Reranker
              |
              v
       Context + Prompt
              |
              v
       Grounded Response

       Observability: LangSmith + Logfire
       Evaluation: RAGAS
```

### Retrieval Pipeline

``` text
User Query
   → Conversation Context
   → Gemini Embedding
   → Qdrant Vector Search
   → Candidate Deduplication
   → Jina Reranking
   → Context Assembly
   → LLM Generation
   → NeMo Guardrails
   → Final Response
```

## Technology Stack

  Area                 Technologies
  -------------------- ------------------------------------
  Backend              Python, FastAPI, Uvicorn, Pydantic
  Orchestration        LangChain, LangGraph
  Embeddings           Gemini Embeddings
  Vector Database      Qdrant
  Reranking            Jina Reranker
  LLM Infrastructure   Portkey
  Guardrails           NVIDIA NeMo Guardrails
  Authentication       Clerk, JWT
  Interface            Streamlit
  Observability        LangSmith, Logfire
  Evaluation           RAGAS
  Deployment           Docker, Render

## Project Structure

``` text
Kubernetes-RAG-ChatBot/
├── app/                  # FastAPI and RAG application
├── evals/                # RAGAS evaluation
├── DATA/                 # Knowledge-base data
├── ui/                   # Streamlit interface
├── .streamlit/           # Streamlit configuration
├── Dockerfile
├── requirements.txt
├── .python-version
└── .gitignore
```

## Run Locally

**Requirements:** Python 3.11, Git, Qdrant, and credentials for the
configured model and supporting services.

``` bash
# Clone the repository
git clone https://github.com/TanviPatidar481/Kubernetes-RAG-ChatBot.git
cd Kubernetes-RAG-ChatBot

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload
```

Run the Streamlit interface in a separate terminal:

``` bash
streamlit run ui/app.py
```

API documentation: <http://127.0.0.1:8000/docs>

Configure the environment variables required by the application before
starting the services. Never commit API keys, JWT secrets, or other
credentials.

## Engineering Notes

-   **Reranking memory:** Local Sentence Transformer-based reranking
    increased memory requirements. The deployed workflow uses the hosted
    Jina Reranker API.
-   **Guardrails overhead:** Local testing showed memory usage
    increasing from approximately 301 MB to 574 MB, with an HTTP peak
    near 595 MB.
-   **Independent observability failures:** The `/query` endpoint could
    return successfully while LangSmith tracing encountered a `401`
    authentication error.

## Future Work

-   Hybrid dense and sparse retrieval.
-   Improved query routing and long-conversation handling.
-   Streaming responses.
-   Expanded evaluation and regression testing.
-   Resource optimization and stronger failure recovery.

## Deployment

-   **API:**
    [kubernetes-rag-chatbot.onrender.com](https://kubernetes-rag-chatbot.onrender.com/)
-   **UI:** [kub-bot-ui.onrender.com](https://kub-bot-ui.onrender.com/)

Availability depends on the hosting environment and external service
configuration.
