"""Answer document questions using retrieved context and selected providers."""

import re

from src.generation_service import generate_text
from src.embedding_service import generate_embedding
from src.vector_store_service import search_chunks


_STOPWORDS = frozenset(
    "how many did i in the a an of to is was and for my "
    "what when where which who why does have has had with from that this "
    "were been are your you about into".split()
)


def _rerank_chunks(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rank candidates by cosine similarity and query-token overlap in memory."""
    query_tokens = {
        token for token in re.findall(r"\w+", question.lower())
        if len(token) >= 4 and token not in _STOPWORDS
    }

    def hybrid_score(chunk: dict) -> float:
        chunk_tokens = set(re.findall(r"\w+", chunk["text"].lower()))
        lexical_overlap = (
            len(query_tokens & chunk_tokens) / len(query_tokens)
            if query_tokens else 0.0
        )
        vector_score = 1 / (1 + chunk["distance"])
        return vector_score + 0.25 * lexical_overlap

    return sorted(chunks, key=hybrid_score, reverse=True)[:top_k]


def answer_question(question: str, top_k: int = 3) -> dict:
    """Generate a cited context-based answer and return labeled sources in order.

    Raise ValueError for blank questions or non-positive top_k. Embedding,
    retrieval, and generation errors propagate to the caller.
    """
    if not question.strip():
        raise ValueError("Question must not be empty or whitespace-only.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    query = generate_embedding(question)
    candidates = search_chunks(query["embedding"], top_k=20)
    chunks = _rerank_chunks(question, candidates, top_k)
    context = "\n\n".join(
        f"[S{index}]\n"
        f"Source: {chunk['source']}\n"
        f"Page: {chunk['page_number']}\n"
        f"Text: {chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )
    system_prompt = (
        "You are a document question-answering assistant. "
        "Answer only using the supplied context. "
        "Every factual claim taken from the context must cite the relevant "
        "source label, such as [S1]. Cite only chunks that support the claim. "
        "Do not invent source labels. "
        "If the context does not contain enough information to answer, "
        "say that the answer could not be found in the provided documents. "
        "Do not use outside knowledge."
    )
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    answer = generate_text(
        prompt,
        system_prompt=system_prompt,
        max_tokens=300,
        temperature=0.0,
    )

    return {
        "answer": answer,
        "sources": [
            {
                "label": f"S{index}",
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "distance": chunk["distance"],
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
    }
