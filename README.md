# Enterprise RAG on AWS

An end-to-end Retrieval-Augmented Generation (RAG) application designed to explore both hands-on GenAI implementation and enterprise AI architecture patterns on **AWS**.

The system supports PDF ingestion, persistent vector search, hybrid retrieval, grounded LLM responses with citations, and configurable model providers. The application includes working integrations with **Amazon Bedrock**, **Amazon Titan Text Embeddings V2**, **Amazon Nova Micro**, **Amazon S3**, **AWS IAM**, and was deployed and validated on **Amazon EC2**.

Local models using BGE/FastEmbed and Llama 3.2/Ollama are also supported for development, testing, and low-cost execution.

## Architecture

### AWS-Enabled RAG Path

```text
                         User / Streamlit UI
                                  │
                                  ▼
                            RAG Pipeline
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
      Amazon Titan Text                    Document Ingestion
       Embeddings V2                              │
      via Amazon Bedrock                         ▼
                 │                        PDF Extraction
                 ▼                          + Chunking
          Chroma Vector DB                         │
                 │                                 ▼
                 ▼                      Amazon Titan Text
        Semantic Retrieval                Embeddings V2
                 │                      via Amazon Bedrock
                 ▼                                 │
        Hybrid Reranking                           ▼
                 │                         Chroma Vector DB
                 ▼
          Grounded Context
                 │
                 ▼
        Amazon Nova Micro
        via Amazon Bedrock
                 │
                 ▼
        Answer + Citations
```

Documents can optionally be stored in **Amazon S3**, with access controlled through **AWS IAM**.

### Local Development Path

```text
BGE / FastEmbed → Chroma → Hybrid Retrieval → Llama 3.2 / Ollama
```

The same RAG pipeline supports both execution modes through provider abstractions.

```text
Embeddings                            Generation

Amazon Titan V2                      Amazon Nova Micro
via Amazon Bedrock                   via Amazon Bedrock
        OR                                   OR
FastEmbed / BGE                      Ollama / Llama 3.2
```

This allows the application to switch between local and **AWS-managed AI services** without rewriting the core RAG logic.

## Key Features

- PDF upload, extraction and overlapping chunking
- deterministic document and chunk IDs using content hashing
- duplicate-safe document ingestion
- **Amazon Titan Text Embeddings V2** integration through **Amazon Bedrock**
- **Amazon Nova Micro** integration through **Amazon Bedrock**
- optional document storage using **Amazon S3**
- access control through **AWS IAM**
- local BGE embeddings using FastEmbed
- persistent vector storage using Chroma
- semantic retrieval with lexical reranking
- local generation using Llama 3.2 3B through Ollama
- grounded answers with `[S1]`, `[S2]`, etc. source citations
- configurable local vs **Amazon Bedrock** providers
- Streamlit document upload and conversational interface
- functional, duplicate, retrieval and failure-path testing
- deployment and validation on **Amazon EC2**

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud platform | **AWS** |
| Foundation model platform | **Amazon Bedrock** |
| AWS embeddings | **Amazon Titan Text Embeddings V2** |
| AWS generation | **Amazon Nova Micro** |
| Document storage | **Amazon S3** |
| Access control | **AWS IAM** |
| Cloud deployment | **Amazon EC2** |
| AWS SDK | **Boto3** |
| Language | Python 3.12 |
| UI | Streamlit |
| PDF processing | pypdf |
| Local embeddings | FastEmbed + BGE |
| Vector database | Chroma |
| Local LLM | Llama 3.2 3B + Ollama |
| Service management | systemd |

## Retrieval Design

The initial implementation used pure semantic vector retrieval. During multi-document testing, a query containing a specific entity retrieved semantically similar chunks from unrelated documents while the correct document appeared much lower in the ranking.

The retrieval pipeline was enhanced to:

```text
Vector retrieval
      ↓
Top candidate pool
      ↓
Vector similarity + lexical overlap
      ↓
Hybrid reranking
      ↓
Top context chunks
```

This improved entity-sensitive retrieval while retaining semantic search for natural-language questions.

The same retrieval layer works independently of whether embeddings are generated locally or through **Amazon Bedrock**.

## Testing & AWS Deployment

The application was validated for:

- correct answers across multiple documents
- grounded answers with source citations
- unsupported-question handling
- duplicate document ingestion
- invalid document handling
- unavailable model-service handling
- missing-file handling
- cross-document retrieval quality
- local and **Amazon Bedrock** provider integration

The full application stack was deployed and tested on **Amazon EC2**:

```text
Amazon EC2
├── Streamlit
├── Python RAG application
├── FastEmbed / BGE
├── Chroma
├── Ollama
└── Llama 3.2 3B
```

The deployed application successfully supported remote document upload, ingestion, retrieval, and cited question answering.

The codebase also retains **Amazon Bedrock** integrations so the same application can use **Titan Text Embeddings V2** and **Nova Micro** instead of local models.

## Production AWS Evolution

The current implementation validates the core RAG architecture. For an enterprise-scale deployment on **AWS**, the same logical design can evolve toward:

```text
Enterprise UI
     ↓
Amazon Cognito
     ↓
Amazon API Gateway
     ↓
AWS Lambda / Amazon ECS on Fargate
     ↓
Amazon S3
     ↓
Amazon Bedrock
     ├── Titan Text Embeddings V2
     └── Foundation Model Generation
     ↓
Amazon OpenSearch Serverless
     ↓
Hybrid Retrieval / Reranking
     ↓
Grounded Answer + Citations
```

In this production architecture:

- documents move to **Amazon S3**
- authentication can use **Amazon Cognito**
- APIs can be exposed through **Amazon API Gateway**
- stateless application logic can run on **AWS Lambda** or **Amazon ECS with AWS Fargate**
- embeddings and generation can run through **Amazon Bedrock**
- vector search can move from Chroma to **Amazon OpenSearch Serverless**
- observability can use **Amazon CloudWatch**
- permissions remain controlled through **AWS IAM**

This keeps the application stateless, scalable, and aligned with managed **AWS** services.

## Project Structure

```text
enterprise-rag-aws/
├── src/
│   ├── bedrock_client.py
│   ├── document_processor.py
│   ├── document_identity.py
│   ├── embedding_service.py
│   ├── generation_service.py
│   ├── ingestion_pipeline.py
│   ├── local_embedding_client.py
│   ├── local_generation_client.py
│   ├── rag_pipeline.py
│   ├── s3_client.py
│   ├── vector_store.py
│   └── vector_store_service.py
├── ui/
│   └── app.py
├── tests/
├── requirements.txt
└── README.md
```

## Run Locally

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the fully local configuration:

```bash
EMBEDDING_PROVIDER=local \
GENERATION_PROVIDER=local \
python -m streamlit run ui/app.py
```

The same application can also use **Amazon Bedrock** providers for embeddings and generation through the configured AWS environment and credentials.