"""Route embedding requests to the configured provider, defaulting to local."""

import os


def get_embedding_provider() -> str:
    """Read EMBEDDING_PROVIDER and reject values other than local or bedrock."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "local")
    if provider not in ("local", "bedrock"):
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER: {provider!r}. "
            "Supported values are 'local' and 'bedrock'."
        )
    return provider


def generate_embedding(text: str) -> dict:
    """Generate an embedding using only the selected provider's lazy import."""
    provider = get_embedding_provider()
    if provider == "local":
        from src.local_embedding_client import generate_local_embedding

        return generate_local_embedding(text)

    from src.bedrock_client import generate_embedding as generate_bedrock_embedding

    return generate_bedrock_embedding(text)
