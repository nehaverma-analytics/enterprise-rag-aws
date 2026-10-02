"""Titan embeddings and Nova Micro text generation through Amazon Bedrock."""

import json
from functools import lru_cache

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError


@lru_cache(maxsize=1)
def _get_bedrock_client() -> BaseClient:
    """Lazily create and reuse the Bedrock Runtime client for rag-app."""
    session = boto3.Session(profile_name="rag-app")
    return session.client("bedrock-runtime", region_name="us-east-1")


def generate_embedding(text: str) -> dict:
    """Return a normalized 1024-dimensional embedding and input token count.

    Raises:
        ValueError: If text is empty or contains only whitespace.
        ClientError: If AWS rejects the request; the original error is preserved.
    """
    if not text.strip():
        raise ValueError("Input text must not be empty or whitespace-only.")

    try:
        client = _get_bedrock_client()
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {"inputText": text, "dimensions": 1024, "normalize": True}
            ),
        )
    except ClientError as error:
        error.add_note("Titan Text Embeddings V2 request failed in us-east-1.")
        raise

    body = response["body"]
    try:
        result = json.loads(body.read())
    finally:
        body.close()

    return {
        "embedding": result["embedding"],
        "input_token_count": result["inputTextTokenCount"],
    }


def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0.0,
) -> str:
    """Generate text with Nova Micro using the Bedrock Converse API.

    Raises:
        ValueError: If prompt is empty or contains only whitespace.
        ClientError: If AWS rejects the request; the original error is preserved.
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty or whitespace-only.")

    request = {
        "modelId": "amazon.nova-micro-v1:0",
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system_prompt is not None:
        request["system"] = [{"text": system_prompt}]

    try:
        client = _get_bedrock_client()
        response = client.converse(**request)
    except ClientError as error:
        error.add_note("Nova Micro text generation request failed in us-east-1.")
        raise

    return "".join(
        block["text"]
        for block in response["output"]["message"]["content"]
        if "text" in block
    )
