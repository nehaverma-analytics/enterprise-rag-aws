"""Route chunk storage and search to the selected embedding collection."""

from src import vector_store
from src.embedding_service import get_embedding_provider


def store_chunks(chunks: list[dict], source_name: str) -> int:
    """Store pre-embedded chunks in the selected provider's collection."""
    if get_embedding_provider() == "local":
        return vector_store.store_local_chunks(chunks, source_name)
    return vector_store.store_chunks(chunks, source_name)


def search_chunks(query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """Search the selected provider's collection with a precomputed embedding."""
    if get_embedding_provider() == "local":
        return vector_store.search_local_similar_chunks(query_embedding, top_k)
    return vector_store.search_similar_chunks(query_embedding, top_k)
