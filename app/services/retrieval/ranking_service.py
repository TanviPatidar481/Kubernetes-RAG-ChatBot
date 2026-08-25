import time
import logfire
from flashrank import Ranker, RerankRequest

# Conservative FlashRank relevance floor. Candidate chunks scoring below this
# threshold are NOT handed to the responder. This filters obvious noise on
# weak / off-topic queries — it deliberately never refuses or blocks a
# question: the responder always answers (with less, or even no, context).
# Tune only after reviewing Logfire's "[Reranker] Top semantic score" logs.
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


# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily. 
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()
    return _ranker



def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    
    Why FlashRank? 
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()
        
        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        
        # Results are returned sorted by highest semantic score first.
        # Relevance floor: keep only chunks scoring at least RELEVANCE_FLOOR,
        # so weak queries return FEWER chunks (possibly none) instead of
        # forcing exactly 5 marginally-related ones onto the responder.
        # This is a context-quality filter only — never an out-of-scope
        # refusal mechanism.
        candidates = results[:top_n]
        reranked_docs = [
            res['text'] for res in candidates
            if res['score'] >= RELEVANCE_FLOOR
        ]

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        if len(reranked_docs) < len(candidates):
            logfire.info(
                f"🎚️ [Reranker] Relevance floor {RELEVANCE_FLOOR}: kept "
                f"{len(reranked_docs)}/{len(candidates)} candidate chunk(s)."
            )

        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]
