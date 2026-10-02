"""Local persistent storage for precomputed Titan and local BGE embeddings."""

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    """Lazily open the persistent cosine collection without an embedding model."""
    client = chromadb.PersistentClient(path="data/chroma")
    return client.get_or_create_collection(
        name="rag_documents",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def store_chunks(chunks: list[dict], source_name: str) -> int:
    """Upsert pre-embedded chunks by chunk_id and return the collection count.

    IDs are collection-wide: an existing chunk_id is replaced on each upsert.
    An empty input leaves records unchanged and returns the current count.
    """
    collection = _get_collection()
    if chunks:
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=[chunk["embedding"] for chunk in chunks],
            metadatas=[
                {
                    "source": source_name,
                    "page_number": chunk["page_number"],
                    "embedding_token_count": chunk["embedding_token_count"],
                }
                for chunk in chunks
            ],
        )
    return collection.count()


def get_collection_count() -> int:
    """Return the number of records in the persistent document collection."""
    return _get_collection().count()


@lru_cache(maxsize=1)
def _get_local_collection() -> Collection:
    """Lazily open the separate cosine collection for local BGE embeddings."""
    client = chromadb.PersistentClient(path="data/chroma")
    return client.get_or_create_collection(
        name="rag_documents_local_bge",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def store_local_chunks(chunks: list[dict], source_name: str) -> int:
    """Upsert pre-embedded BGE chunks and return the local collection count.

    An empty input leaves records unchanged and returns the current count.
    """
    collection = _get_local_collection()
    if chunks:
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=[chunk["embedding"] for chunk in chunks],
            metadatas=[
                {
                    "source": source_name,
                    "page_number": chunk["page_number"],
                    "embedding_model": "BAAI/bge-small-en-v1.5",
                }
                for chunk in chunks
            ],
        )
    return collection.count()


def get_local_collection_count() -> int:
    """Return the number of records in the local BGE collection."""
    return _get_local_collection().count()


def search_similar_chunks(
    query_embedding: list[float], top_k: int = 3
) -> list[dict]:
    """Return up to top_k matches in Chroma's distance order.

    Raise ValueError for an empty embedding or non-positive top_k. Return an
    empty list when the collection has no records.
    """
    if not query_embedding:
        raise ValueError("query_embedding must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": chunk_id,
            "text": results["documents"][0][index],
            "source": results["metadatas"][0][index]["source"],
            "page_number": results["metadatas"][0][index]["page_number"],
            "distance": results["distances"][0][index],
        }
        for index, chunk_id in enumerate(results["ids"][0])
    ]


def search_local_similar_chunks(
    query_embedding: list[float], top_k: int = 3
) -> list[dict]:
    """Return local BGE matches in Chroma order, or [] for an empty collection.

    Raise ValueError for an empty embedding or non-positive top_k.
    """
    if not query_embedding:
        raise ValueError("query_embedding must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    collection = _get_local_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": chunk_id,
            "text": results["documents"][0][index],
            "source": results["metadatas"][0][index]["source"],
            "page_number": results["metadatas"][0][index]["page_number"],
            "distance": results["distances"][0][index],
        }
        for index, chunk_id in enumerate(results["ids"][0])
    ]
