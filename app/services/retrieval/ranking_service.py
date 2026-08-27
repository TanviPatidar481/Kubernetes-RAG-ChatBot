import time

import logfire
import requests

from app.config import settings

# Relevance floor applied to Jina's relevance_score (0–1 scale). Candidate
# chunks scoring below this threshold are NOT handed to the responder. This
# filters obvious noise on weak / off-topic queries — it deliberately never
# refuses or blocks a question: the responder always answers (with less, or
# even no, context).
#
# NOTE: recalibrate after the first live runs — Jina's score distribution
# differs from FlashRank's. Watch Logfire's "[Reranker]" logs and adjust.
RELEVANCE_FLOOR = 0.1


def _normalize_for_dedup(text: str) -> str:
    """
    Conservative normalization for exact-content deduplication.

    Only strips surrounding whitespace and normalizes CRLF to LF. We do NOT
    lowercase or collapse internal whitespace, so genuinely different chunks
    (code blocks, YAML with case-sensitive values, URLs, etc.) are never merged.
    """
    if not text:
        return ""
    return text.strip().replace("\r\n", "\n")


def deduplicate_documents(documents: list[dict]) -> tuple[list[dict], int]:
    """
    Removes exact-duplicate chunks based on normalized CONTENT/TEXT, keeping the
    first (best-scoring / original-order) occurrence and its metadata.

    Different chunks from the same source file are preserved — only byte-identical
    normalized content is collapsed to a single candidate.

    Returns:
        (deduplicated_docs, removed_count)
    """
    seen = set()
    deduped = []
    for doc in documents:
        key = _normalize_for_dedup(doc.get("content", "") if isinstance(doc, dict) else str(doc))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    removed = len(documents) - len(deduped)
    if removed:
        logfire.info(
            f"♻️ Deduplicated {removed} exact-content duplicate(s): "
            f"{len(documents)} → {len(deduped)} candidates."
        )
    return deduped, removed


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query
    semantically using the Jina Reranker API (hosted cross-encoder).

    Why hosted instead of local ONNX?
    Standard vector search (Cosine Similarity) is fast but mathematically
    "fuzzy." A cross-encoder is far more precise — but running it locally via
    FlashRank loaded model weights + an ONNX runtime session into our process
    (~250–300 MB at peak). Delegating to Jina removes that entire footprint
    while keeping the same ranking quality contract: results come back sorted
    best-first, ready for the relevance-floor filter.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to Jina Reranker...")

    try:
        api_key = settings.JINA_API_KEY
        if not api_key:
            raise ValueError("JINA_API_KEY is not configured")

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        # Results arrive sorted best-first; each entry maps back into our
        # input order via 'index'.
        scored: list[tuple[str, float]] = []
        for hit in payload.get("results", []):
            idx = hit.get("index")
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scored.append((documents[idx], float(hit.get("relevance_score", 0.0))))

        # Relevance floor: keep only chunks scoring at least RELEVANCE_FLOOR,
        # so weak queries return FEWER chunks (possibly none) instead of
        # forcing exactly `top_n` marginally-related ones onto the responder.
        # This is a context-quality filter only — never an out-of-scope
        # refusal mechanism.
        reranked_docs = [text for text, score in scored if score >= RELEVANCE_FLOOR]

        duration = time.time() - start_time
        top_score = scored[0][1] if scored else "N/A"
        logfire.info(
            f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}"
        )
        if len(reranked_docs) < len(scored):
            logfire.info(
                f"🎚️ [Reranker] Relevance floor {RELEVANCE_FLOOR}: kept "
                f"{len(reranked_docs)}/{len(scored)} candidate chunk(s)."
            )

        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]