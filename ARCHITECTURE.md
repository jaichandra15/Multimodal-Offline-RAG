# System Architecture

## RAG Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Query Embedding (Ollama)                          │
│                   nomic-embed-text (768-dim)                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Hybrid Search                                   │
│                                                                      │
│  Vector Search (cosine similarity)                                  │
│  +                                                                   │
│  Keyword Search (PostgreSQL full-text search)                       │
│                                                                      │
│  Returns: Top 30 candidates                                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Cross-Encoder Reranking                            │
│                                                                      │
│  Model: cross-encoder/ms-marco-MiniLM-L-6-v2                        │
│  Input: [query, chunk_text] pairs                                   │
│  Output: Relevance scores (0-1)                                     │
│                                                                      │
│  Reranks 30 → Selects Top 5 most relevant                           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Context Assembly                                │
│                                                                      │
│  Format: "Source: {filename} (Page {page})\n{chunk_text}\n\n"       │
│  Includes: similarity_score for citation ranking                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Prompt Construction                               │
│                                                                      │
│  System Prompt + Retrieved Context + User Query                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LLM Generation (Ollama)                            │
│                                                                      │
│  Model: mistral (default)                                           │
│  Streaming: Server-Sent Events (SSE)                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Response to User                                │
│                                                                      │
│  - Generated answer (streaming)                                     │
│  - Source citations with page numbers                               │
│  - Similarity scores for ranking                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Why Reranking Improves Quality

### Problem with Hybrid Search Alone

Embedding models are trained to match **semantic similarity** but may not capture:
- Exact keyword importance
- Query intent specificity
- Contextual relevance

This can cause less relevant chunks to rank higher than truly relevant ones.

### How Cross-Encoders Help

**Bi-Encoder (Embedding Model):**
- Encodes query and documents separately
- Fast: O(n) comparisons with pre-computed embeddings
- Good for initial retrieval from large corpus

**Cross-Encoder (Reranker):**
- Encodes query-document **pairs together**
- Slow: O(n) forward passes through transformer
- Better accuracy: sees full interaction between query and document

### The Two-Stage Strategy

1. **Stage 1 (Fast)**: Hybrid search retrieves 30 candidates
   - Uses pre-computed embeddings (fast)
   - Broad net to avoid missing relevant docs

2. **Stage 2 (Accurate)**: Reranker rescores 30 candidates
   - Cross-encoder evaluates each query-chunk pair
   - Selects best 5 for final context
   - Only 30 pairs to score (acceptable latency)

**Result**: 10-20% improvement in answer relevance with +50-200ms latency

## Component Interactions

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                           │
│                                                                      │
│  - Chat UI with streaming                                           │
│  - Metrics dashboard (Recharts)                                     │
│  - Document management                                              │
│  - Zustand state management                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP/SSE
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                                 │
│                                                                      │
│  Routes (api/routes.py)                                             │
│  ├── POST /chat         - RAG conversation                          │
│  ├── POST /upload       - File upload                               │
│  ├── GET /documents     - List docs                                 │
│  ├── GET /health        - System status                             │
│  └── GET /metrics       - Prometheus endpoint                       │
│                                                                      │
│  Middleware                                                          │
│  ├── RequestIDMiddleware - Unique request tracking                  │
│  └── PrometheusMiddleware - HTTP metrics                            │
└────────────────────┬──────────────────┬─────────────────────────────┘
                     │                  │
        ┌────────────▼─────────┐       │
        │   RAG Engine         │       │
        │   (rag_engine.py)    │       │
        │                      │       │
        │  - Search           │       │
        │  - Rerank           │       │
        │  - Generate         │       │
        └──┬────────┬─────────┘       │
           │        │                  │
           │        │                  │
   ┌───────▼──┐  ┌──▼────────┐  ┌─────▼──────┐
   │ Reranker │  │  Ollama   │  │  Database  │
   │          │  │  Client   │  │  (Async)   │
   │ CrossEnc │  │           │  │            │
   │ -oder    │  │ - Chat    │  │ - Vectors  │
   │          │  │ - Embed   │  │ - Chunks   │
   └──────────┘  └───────────┘  └────┬───────┘
                                      │
                                      │
                              ┌───────▼────────┐
                              │  PostgreSQL    │
                              │  + PGVector    │
                              │                │
                              │  - Documents   │
                              │  - Chunks      │
                              │  - Embeddings  │
                              └────────────────┘
```

## Observability Stack

```
┌────────────────────────────────────────────────────────────┐
│                    Request Flow                             │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │  RequestIDMiddleware     │
           │  - Generate UUID         │
           │  - Set context var       │
           │  - Add response headers  │
           └──────────┬───────────────┘
                      │
                      ▼
           ┌──────────────────────────┐
           │  CustomJsonFormatter     │
           │  - timestamp             │
           │  - request_id            │
           │  - level                 │
           │  - message               │
           │  - source (file:line)    │
           └──────────┬───────────────┘
                      │
                      ▼
           ┌──────────────────────────┐
           │  Business Logic          │
           │  - RAG operations        │
           │  - Metric recording      │
           └──────────┬───────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌────────────────┐       ┌─────────────────┐
│  JSON Logs     │       │  Prometheus     │
│  (stdout)      │       │  Metrics        │
│                │       │                 │
│  - Structured  │       │  Histograms:    │
│  - Searchable  │       │  - latencies    │
│  - Traceable   │       │  - rank_change  │
└────────────────┘       │                 │
                         │  Counters:      │
                         │  - requests     │
                         │  - calls        │
                         │  - errors       │
                         └─────────┬───────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │  Frontend       │
                         │  Dashboard      │
                         │  (Recharts)     │
                         └─────────────────┘
```

## Data Flow: Document Ingestion

```
┌───────────────┐
│  Upload File  │
│  (Frontend)   │
└───────┬───────┘
        │
        ▼
┌──────────────────────────┐
│  POST /upload            │
│  (FastAPI endpoint)      │
└───────┬──────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Ingestion Pipeline      │
│  (pipeline.py)           │
│                          │
│  1. File validation      │
│  2. Format detection     │
│  3. Content extraction   │
└───────┬──────────────────┘
        │
        ├─ Audio? ──────────┐
        │                   ▼
        │         ┌──────────────────┐
        │         │  Whisper ASR     │
        │         │  (transcription) │
        │         └─────────┬────────┘
        │                   │
        ▼                   │
┌──────────────────────────┼──┐
│  Docling Chunker         │  │
│  (chunker.py)            │  │
│                          ◄──┘
│  - HybridChunker         │
│  - Semantic boundaries   │
│  - Max 1000 tokens       │
│  - Overlap 100 tokens    │
└───────┬──────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Embedder                │
│  (embedder.py)           │
│                          │
│  - Ollama API call       │
│  - nomic-embed-text      │
│  - 768-dim vectors       │
└───────┬──────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Database Operations     │
│  (operations.py)         │
│                          │
│  - Insert Document       │
│  - Insert Chunks         │
│  - Store vectors         │
└──────────────────────────┘
```

## Configuration Hierarchy

```
Environment Variables (.env)
           │
           ▼
┌──────────────────────────┐
│  config.py (Settings)    │
│                          │
│  - Pydantic validation   │
│  - Type safety           │
│  - Default values        │
└───────┬──────────────────┘
        │
        ├──► Database config
        │    - Connection string
        │    - Pool settings
        │
        ├──► Ollama config
        │    - Base URL
        │    - Models
        │    - Timeout
        │
        ├──► RAG config
        │    - top_k chunks
        │    - hybrid weights
        │    - reranker settings
        │
        └──► Reranker config
             - Enabled flag
             - Model name
             - Top K candidates
             - Batch size
```

## Performance Characteristics

| Component | Latency | Bottleneck | Optimization |
|-----------|---------|------------|--------------|
| **Embedding** | 50-100ms | Ollama API | Use GPU Ollama |
| **Vector Search** | 10-50ms | Database | Add indexes |
| **Keyword Search** | 5-20ms | Database | GIN index |
| **Reranking** | 50-200ms | GPU/CPU | Batch inference |
| **LLM Generation** | 2-10s | Ollama API | Streaming |

**Total Query Latency**: 2.1-10.5 seconds

**Breakdown**:
- Search: 200-300ms (with reranker)
- Generation: 2-10s (streaming starts immediately)
