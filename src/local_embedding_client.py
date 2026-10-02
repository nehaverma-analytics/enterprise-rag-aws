"""Local embeddings using FastEmbed and BAAI's small English BGE model."""

from threading import Lock

from fastembed import TextEmbedding


MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: TextEmbedding | None = None
_model_lock = Lock()


def _get_model() -> TextEmbedding:
    """Lazily load and reuse one model instance within this process."""
    global _model
    with _model_lock:
        if _model is None:
            _model = TextEmbedding(model_name=MODEL_NAME)
        return _model


def generate_local_embedding(text: str) -> dict:
    """Return one local embedding as a list and its model name.

    Raises ValueError if text is empty or contains only whitespace.
    """
    if not text.strip():
        raise ValueError("Input text must not be empty or whitespace-only.")

    model = _get_model()
    embedding = next(iter(model.embed([text])))

    return {
        "embedding": embedding.tolist(),
        "model": MODEL_NAME,
    }
