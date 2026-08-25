import re

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
    RAIL_INDICATORS,
    BLOCK_PATTERNS,
    BLOCK_REFUSALS,
    REFUSAL_HEURISTICS,
)


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses openai/gpt-oss-20b for fast intent classification at the gate —
    the heavier openai/gpt-oss-120b is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (openai/gpt-oss-20b).")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the guardrails gate.

    Layer 1 — deterministic pattern pre-filter (no LLM call).
    Layer 2 — NeMo LLMRails intent classification.
    Layer 3 — conservative safety-refusal signals (explicit jailbreak/safety
              refusals only; a generic "can't help" is NOT treated as a fire).

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    # ---------------- Layer 1: deterministic pre-filter ----------------
    lowered = (message or "").lower()
    for pattern, category in BLOCK_PATTERNS:
        if re.search(pattern, lowered):
            logfire.info(
                f"🛡️ Rule matched [{category}] pattern={pattern} | query='{message[:80]}'"
            )
            return True, BLOCK_REFUSALS[category]

    # ---------------- Layer 2: NeMo LLM rails ----------------
    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        content = content or ""

        # ---------------- Layer 3: refusal heuristics ----------------
        fired = any(indicator in content for indicator in RAIL_INDICATORS)
        if not fired:
            lowered_content = content.lower()
            fired = any(h in lowered_content for h in REFUSAL_HEURISTICS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
