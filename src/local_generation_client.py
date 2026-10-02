"""Generate text through the locally running Ollama HTTP API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_local_text(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0.0,
) -> str:
    """Return generated text, raising ValueError for a blank prompt.

    Raise RuntimeError for connection failures or invalid Ollama responses.
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty or whitespace-only.")

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if system_prompt is not None:
        payload["system"] = system_prompt

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except HTTPError as error:
        with error:
            details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama request failed (HTTP {error.code}): {details or error.reason}"
        ) from error
    except (URLError, OSError) as error:
        raise RuntimeError(
            f"Unable to reach Ollama at {OLLAMA_URL}. "
            f"Check that Ollama is running and reachable. Details: {error}"
        ) from error
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f"Ollama returned invalid JSON: {error}") from error

    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned an invalid response object.")
    if result.get("error"):
        raise RuntimeError(f"Ollama generation failed: {result['error']}")
    if not isinstance(result.get("response"), str):
        raise RuntimeError("Ollama response is missing generated text.")
    return result["response"]
