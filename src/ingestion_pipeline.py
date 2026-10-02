"""Ingest PDFs using stable document IDs and the selected embedding provider."""

from src.document_identity import generate_document_id
from src.document_processor import chunk_pages, extract_pdf_text
from src.embedding_service import generate_embedding
from src.vector_store_service import store_chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return new enriched dictionaries for non-empty chunks, preserving metadata.

    Validate all required fields before embedding. Raise ValueError for missing
    fields and TypeError for non-string text. Skip whitespace-only text and
    propagate embedding errors without modifying the input chunks.
    """
    required_fields = {"chunk_id", "page_number", "text"}
    for index, chunk in enumerate(chunks):
        missing_fields = required_fields.difference(chunk)
        if missing_fields:
            raise ValueError(
                f"Chunk at index {index} is missing required fields: "
                f"{', '.join(sorted(missing_fields))}"
            )
        if not isinstance(chunk["text"], str):
            raise TypeError(f"Chunk at index {index} must have string text.")

    enriched_chunks: list[dict] = []
    for chunk in chunks:
        if not chunk["text"].strip():
            continue
        result = generate_embedding(chunk["text"])
        enriched_chunk = {**chunk, "embedding": result["embedding"]}
        if "input_token_count" in result:
            enriched_chunk["embedding_token_count"] = result["input_token_count"]
        if "model" in result:
            enriched_chunk["embedding_model"] = result["model"]
        enriched_chunks.append(enriched_chunk)

    return enriched_chunks


def ingest_document(
    local_path: str, original_name: str, s3_bucket: str | None = None
) -> dict:
    """Index a PDF and optionally upload it to S3; propagate processing errors."""
    document_id = generate_document_id(local_path)
    s3_uri = None
    if s3_bucket is not None:
        from src.s3_client import upload_document

        s3_uri = upload_document(
            local_path, s3_bucket, f"documents/{document_id}.pdf"
        )

    pages = extract_pdf_text(local_path)
    chunks = chunk_pages(pages, document_id=document_id)
    embedded_chunks = embed_chunks(chunks)
    records = store_chunks(embedded_chunks, source_name=original_name)
    return {
        "document_id": document_id,
        "original_name": original_name,
        "pages": len(pages),
        "chunks": len(embedded_chunks),
        "records_in_collection": records,
        "s3_uri": s3_uri,
    }
