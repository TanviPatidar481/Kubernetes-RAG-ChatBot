import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.

    Returns:
        list[dict]: list matching points and text chunks (NO_RESULTS if the
        search genuinely returns zero candidates).

    Raises:
        Propagates embedding / Qdrant exceptions (RETRIEVAL_ERROR) so callers
        can distinguish a real search failure from an empty result set.
    """
    query_vector = embed_query(query)

    # Using query_points - the modern standard for Qdrant
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True # JSON
    )

    results = []
    for res in response.points:
        results.append({
            "content": res.payload.get("text", ""),
            "source": res.payload.get("source", "Unknown"),
            "score": res.score
        })

    return results
