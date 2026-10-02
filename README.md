# AWS RAG Application

A planned cost-effective, end-to-end Retrieval-Augmented Generation (RAG) application using Python 3.12. This repository currently contains only the project scaffold; all Python modules are empty placeholders.

## Planned architecture

Documents stored in Amazon S3 will be extracted with pypdf and split into chunks. Amazon Titan Text Embeddings V2 on Amazon Bedrock will embed those chunks for storage in a local persistent Chroma vector database. At query time, the pipeline will retrieve relevant chunks and use Amazon Nova Micro on Bedrock to generate a grounded answer for the Streamlit interface. Boto3 will provide AWS integration.

The planned AWS region is `us-east-1`, using the existing AWS CLI profile `rag-app`, which assumes a restricted development role. Credentials are not stored in this project.

## Project structure

```text
aws-rag-app/
├── src/
│   ├── __init__.py
│   ├── bedrock_client.py
│   ├── document_processor.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── ui/
│   └── app.py
├── tests/
│   └── __init__.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Planned development stages

1. Initialize the local project scaffold (current stage).
2. Implement S3 document loading, PDF extraction, and chunking.
3. Add Bedrock embeddings and persistent Chroma indexing.
4. Build retrieval and answer generation with Nova Micro.
5. Add the Streamlit interface.
6. Test retrieval quality, AWS integration, and cost-conscious usage.

Dependencies are listed in `requirements.txt` but have not been installed. The existing `.venv` is left unchanged. No AWS operations, model invocations, cloud provisioning, or deployment are part of this initialization.
