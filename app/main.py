# ============================================================
# CRITICAL: Configure Logfire before importing app modules
# ============================================================

import os
import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# ============================================================
# Imports
# ============================================================

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional

from app.agents.graph import rag_agent

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="Enterprise Agentic RAG API")

# ============================================================
# Request Model
# ============================================================

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"

# ============================================================
# Routes
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Enterprise LangGraph RAG API is live."
    }


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the LangGraph workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {
            "error": f"Could not generate graph image: {e}"
        }


@app.post("/query")
def query(request: QueryRequest):

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": request.q
            }
        ],
        "current_query": request.q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    try:

        final_output = rag_agent.invoke(
            initial_state,
            config=config
        )

        return {
            "question": request.q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }

    except Exception as e:

        logfire.exception("Backend execution failed")

        return {
            "question": request.q,
            "answer": "I encountered an internal error while processing your request.",
            "thought_process": ["Execution failed"],
            "status": "error",
            "sources": [],
            "error": str(e)
        }