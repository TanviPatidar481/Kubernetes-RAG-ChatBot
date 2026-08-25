import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
GEMINI_MODEL = "models/gemini-embedding-2-preview"
GEMINI_DIM = 3072

# Ordered list of (label, embedder) — one per API key. Every key uses the SAME
# Gemini model, so vectors are always 3072-dim and the locked Qdrant collection
# is never incompatible. There is deliberately NO 768-dim sentence-transformers
# fallback in the production embedding path.
_embedders: list[tuple[str, "GoogleGenerativeAIEmbeddings"]] = []
_sticky_index = 0


# ── Model initialisation ───────────────────────────────────────────────────────

def _is_transient(exc: Exception) -> bool:
    """Return True only for transient/retryable API errors (429/503/rate/quota/timeout)."""
    err = str(exc).lower()
    return any(
        marker in err
        for marker in (
            "429", "503", "rate", "quota", "resource_exhausted",
            "unavailable", "temporarily", "timeout", "connection",
        )
    )


def _init():
    """
    Build the key-rotation embedding list once per process (lazy).

    Both configured keys use the SAME Gemini embedding model, so every vector is
    3072-dimensional and remains compatible with the existing Qdrant collection.
    There is deliberately no 768-dim sentence-transformers fallback.
    """
    global _embedders
    if _embedders:
        return

    primary = settings.GEMINI_API_KEY_PRIMARY or settings.GEMINI_API_KEY
    secondary = settings.GEMINI_API_KEY_SECONDARY

    candidates = [("primary", primary)]
    if secondary:
        candidates.append(("secondary", secondary))

    embedders = []
    for label, key in candidates:
        if not key:
            continue
        embedders.append(
            (
                label,
                GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-2-preview",
                    google_api_key=key,
                ),
            )
        )

    if not embedders:
        raise ValueError(
            "No Gemini embedding key configured. Set GEMINI_API_KEY_PRIMARY "
            "(or GEMINI_API_KEY / GEMINI_API_KEY_SECONDARY) in the environment."
        )

    _embedders[:] = embedders
    logfire.info(
        f"Gemini embeddings ready: models/gemini-embedding-2-preview, "
        f"{GEMINI_DIM}-dim, {len(_embedders)} key(s) for rotation."
    )


def _rotate(embed_fn):
    """
    Bounded key-rotation / failover loop. Tries the last-known-good key first.

    - Transient errors (429/503/rate/quota/timeout/connection) rotate to the
      next key immediately.
    - Permanent errors (invalid key / 400 / 403) raise immediately.
    - After 3 rounds across all keys, raise a clear error.
    - Never returns an empty vector; never downgrades to a 768-dim model.
    """
    global _sticky_index
    errors: list[str] = []

    for attempt in range(3):
        for offset in range(len(_embedders)):
            idx = (_sticky_index + offset) % len(_embedders)
            label, model = _embedders[idx]
            try:
                result = embed_fn(model)
                _sticky_index = idx  # stick to the key that just worked
                if offset > 0:
                    logfire.info(f"Gemini key rotation: failover to [{label}] succeeded.")
                return result
            except Exception as e:
                if not _is_transient(e):
                    logfire.error(f"Gemini [{label}] permanent error: {e}")
                    raise
                logfire.warning(f"Gemini [{label}] transient failure ({e}); trying next key.")
                errors.append(f"{label}: {type(e).__name__}")

        wait = 2 ** attempt
        logfire.warning(
            f"All {len(_embedders)} Gemini key(s) failed this round ({errors[-2:]}); "
            f"backing off {wait}s."
        )
        if attempt < 2:
            time.sleep(wait)

    raise RuntimeError(
        f"Gemini embedding failed after {len(_embedders)} key(s) × 3 rounds: {errors}"
    )


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Qdrant is locked at 3072 dimensions — always return GEMINI_DIM."""
    return GEMINI_DIM


# ── Batch embedding (key rotation) ────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    _init()
    return _rotate(lambda model: model.embed_documents(batch))


# ── Public API (same signatures as before) ────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Embed a single query with bounded key rotation / failover.

    Returns a 3072-dim vector. Raises on permanent errors and after all keys
    fail; never returns an empty vector and never downgrades to a 768-dim
    sentence-transformers model.
    """
    _init()
    return _rotate(lambda model: model.embed_query(query))


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model="gemini-embedding-2-preview", start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
