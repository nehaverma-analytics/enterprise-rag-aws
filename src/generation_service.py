"""Route text generation to the configured provider, defaulting to local."""

import os


def get_generation_provider() -> str:
    """Read GENERATION_PROVIDER and reject unsupported provider values."""
    provider = os.environ.get("GENERATION_PROVIDER", "local")
    if provider not in ("local", "bedrock"):
        raise ValueError(
            f"Unsupported GENERATION_PROVIDER: {provider!r}. "
            "Supported values are 'local' and 'bedrock'."
        )
    return provider


def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0.0,
) -> str:
    """Lazily load the selected provider and forward generation parameters."""
    provider = get_generation_provider()
    if provider == "local":
        from src.local_generation_client import generate_local_text

        return generate_local_text(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    from src.bedrock_client import generate_text as generate_bedrock_text

    return generate_bedrock_text(
        prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
