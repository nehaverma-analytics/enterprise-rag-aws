"""Page-by-page PDF text extraction and character-based chunking."""

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: str) -> list[dict]:
    """Return trimmed text for every page, numbered from 1, including empty pages.

    Raises FileNotFoundError if the path is not an existing file and ValueError
    if its extension is not .pdf (case-insensitive). PDF reading errors propagate.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("File must have a .pdf extension.")

    with path.open("rb") as document:
        reader = PdfReader(document)
        return [
            {
                "page_number": page_number,
                "text": (page.extract_text() or "").strip(),
            }
            for page_number, page in enumerate(reader.pages, start=1)
        ]


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200,
    document_id: str | None = None,
) -> list[dict]:
    """Split each page into trimmed, overlapping character windows.

    Preserve page metadata and number chunks from 1 within each page. Skip
    empty or whitespace-only chunks. Raise ValueError for invalid sizes.
    Prefix chunk IDs with document_id when supplied; otherwise keep legacy IDs.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size.")

    chunks: list[dict] = []
    step = chunk_size - overlap
    for page in pages:
        text = page["text"]
        if not text.strip():
            continue

        chunk_number = 1
        for start in range(0, len(text), step):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"page_{page['page_number']}_chunk_{chunk_number}"
                if document_id is not None:
                    chunk_id = f"{document_id}_{chunk_id}"
                chunks.append(
                    {
                        **page,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )
                chunk_number += 1
            if end >= len(text):
                break

    return chunks
