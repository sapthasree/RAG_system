# Production-Ready Retrieval-Augmented Generation (RAG) System for Text Documents

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that ingests raw text documents, preprocesses and chunks them, generates semantic embeddings, stores them in a **FAISS vector database**, and retrieves relevant context to generate grounded answers using a Large Language Model.

The system is designed with **modularity**, **persistence**, and **debuggability** in mind, mimicking real-world AI research and engineering workflows.

## Architecture

```mermaid
flowchart TD
    A[Raw Text] --> B[Ingestion]
    B --> C[Preprocessing]
    C --> D[Chunking with Overlap]
    D --> E[Embedding Generation]
    E --> F[FAISS Vector Store (Persistent)]
    F --> G[Semantic Retrieval (Top-K)]
    G --> H[Context Construction]
    H --> I[LLM-based Answer Generation]
```

## Key Features
- Modular ETL pipeline (ingestion → preprocessing → chunking)
- Semantic embeddings using SentenceTransformers
- Fast similarity search with FAISS
- Persistent vector index to avoid recomputation
- Retrieval-augmented LLM responses with hallucination control
- Config-driven hyperparameter tuning
- Logging and basic retrieval evaluation

## Tech Stack
- **Language**: Python
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS
- **LLM**: OpenAI API (configurable)
- **Utilities**: NumPy, Logging

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Add your text data
Place your document inside:
```
data/sample.txt
```

### 3. Run the pipeline
```
python main.py
```

### 4. Ask questions interactively
The system retrieves relevant document chunks and generates answers grounded strictly in the retrieved context.

## Evaluation & Debugging
- Retrieved chunks are inspected before generation to ensure relevance
- Logging is enabled for monitoring pipeline stages
- Guardrails prevent hallucinated responses when answers are not present in the data

## Known Limitations
- Designed for text documents (PDF OCR not included)
- Flat FAISS index (can be extended to ANN for large-scale datasets)
- Single-document ingestion (multi-document ingestion can be added)

## Future Improvements
- Support for PDFs and web data
- ANN indexing (IVF/HNSW)
- Metadata filtering
- Multi-document and multi-domain ingestion
- Local LLM integration

## Author
Sapthasree N K
