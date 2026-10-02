"""Minimal Streamlit chat interface for indexed RAG documents."""

import logging
import os
from pathlib import Path
import sys
from uuid import uuid4

import streamlit as st

# Allow `streamlit run ui/app.py` to resolve the project's src package.
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion_pipeline import ingest_document
from src.rag_pipeline import answer_question


logger = logging.getLogger(__name__)


def display_message(message: dict) -> None:
    """Render one saved chat message and its source references."""
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.text(
                        f"[{source['label']}] {source['source']}\n"
                        f"Page: {source['page_number']}\n"
                        f"Chunk ID: {source['chunk_id']}"
                    )
                if not message["sources"]:
                    st.caption("No sources were retrieved.")


st.set_page_config(page_title="AWS RAG Document Assistant")
st.title("AWS RAG Document Assistant")
st.caption("Ask questions about the documents currently indexed in the RAG system.")

with st.sidebar:
    st.subheader("RAG Configuration")
    st.caption("Embeddings: Local BGE")
    st.caption("Generation: Local Llama 3.2")
    st.caption("Vector store: Local Chroma")

    st.header("Documents")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        max_upload_size=20,
    )
    if uploaded_pdf is not None:
        st.text(uploaded_pdf.name)
        st.caption(f"Size: {uploaded_pdf.size / (1024 * 1024):.2f} MB")
        if st.button("Prepare document"):
            try:
                uploads_dir = Path(project_root) / "data" / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                local_path = uploads_dir / f"{uuid4().hex}.pdf"
                with local_path.open("xb") as destination:
                    destination.write(uploaded_pdf.getvalue())
            except OSError:
                logger.exception("Unable to save the uploaded PDF.")
                st.error("Unable to save the PDF. Please try again.")
            else:
                st.session_state["uploaded_original_name"] = uploaded_pdf.name
                st.session_state["uploaded_local_path"] = str(local_path)
                st.session_state.pop("last_ingestion_result", None)
                st.success("Document saved locally and ready for processing.")

    prepared_path = st.session_state.get("uploaded_local_path")
    original_name = st.session_state.get("uploaded_original_name")
    if prepared_path and original_name and Path(prepared_path).is_file():
        if st.button("Ingest document"):
            st.session_state.pop("last_ingestion_result", None)
            try:
                with st.spinner("Ingesting document..."):
                    ingestion_result = ingest_document(
                        prepared_path,
                        original_name,
                        s3_bucket=os.environ.get("RAG_S3_BUCKET", "").strip() or None,
                    )
            except Exception:
                logger.exception("Unable to ingest the prepared document.")
                st.error("Unable to ingest the document. Please try again.")
            else:
                st.session_state["last_ingestion_result"] = ingestion_result
                st.success("Document ingested successfully.")

    if ingestion_result := st.session_state.get("last_ingestion_result"):
        st.text(f"Document ID: {ingestion_result['document_id']}")
        st.text(f"Pages processed: {ingestion_result['pages']}")
        st.text(f"Chunks indexed: {ingestion_result['chunks']}")
        st.text(f"Records in vector collection: {ingestion_result['records_in_collection']}")
        if ingestion_result["s3_uri"]:
            st.text(f"S3 URI: {ingestion_result['s3_uri']}")
        else:
            st.caption("Stored locally only")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    display_message(message)

if question := st.chat_input("Ask a question about your documents"):
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    display_message(user_message)

    try:
        with st.spinner("Searching documents and generating answer..."):
            result = answer_question(question, top_k=3)
        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    except RuntimeError as error:
        logger.exception("Unable to generate the document answer.")
        assistant_message = {
            "role": "assistant",
            "content": "Unable to generate an answer. For local generation, check "
            "that Ollama is running and llama3.2:3b is available. "
            f"Details: {error}",
            "error": True,
        }
    except Exception:
        logger.exception("Unable to answer the document question.")
        assistant_message = {
            "role": "assistant",
            "content": "Sorry, I couldn't answer your question. Please try again. "
            "If the problem continues, check the application logs.",
            "error": True,
        }

    st.session_state.messages.append(assistant_message)
    display_message(assistant_message)
