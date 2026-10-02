"""Stable document identifiers derived from file contents."""

import hashlib
from pathlib import Path


def generate_document_id(file_path: str) -> str:
    """Return the first 12 SHA-256 hex characters, reading the file in blocks.

    Raise FileNotFoundError if the path is not an existing file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    digest = hashlib.sha256()
    with path.open("rb") as document:
        for block in iter(lambda: document.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]
