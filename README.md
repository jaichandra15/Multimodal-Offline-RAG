# Multimodal RAG System with Ollama

A production-ready **Retrieval-Augmented Generation (RAG)** system with full-stack implementation: FastAPI backend, Next.js frontend, PostgreSQL/PGVector, and Ollama for fully local/offline LLM inference.

## 🌟 Features

### Backend
- ✅ **Ollama-based** - fully offline LLM inference
- ✅ **FastAPI Backend** - RESTful API with Server-Sent Events (SSE) streaming
- ✅ **SQLAlchemy ORM** - Async database operations with clean architecture
- ✅ **PGVector Integration** - Semantic search with 768-dim embeddings
- ✅ **Cross-Encoder Reranking** - Improved relevance with sentence-transformers
- ✅ **Multi-format Support** - PDF, Word, Excel, PowerPoint, Markdown, Audio
- ✅ **Hybrid Chunking** - Intelligent document splitting with Docling
- ✅ **Audio Transcription** - Whisper ASR for MP3/WAV/M4A/FLAC files
- ✅ **Citation Tracking** - Source attribution with page numbers
- ✅ **Observability** - Structured JSON logging + Prometheus metrics

### Frontend
- ✅ **Next.js 14** - React with App Router and TypeScript
- ✅ **Real-time Streaming** - Server-Sent Events for live LLM responses
- ✅ **Modern UI** - shadcn/ui components with Tailwind CSS
- ✅ **Chat Interface** - ChatGPT-style conversation experience
- ✅ **Document Management** - Upload, view, and manage knowledge base
- ✅ **Citation Display** - Interactive source references with expand/collapse
- ✅ **Conversation History** - Persistent chat sessions with rename/delete
- ✅ **Metrics Dashboard** - Real-time performance monitoring with Recharts
- ✅ **Dark Theme** - Optimized for readability

## 📁 Project Structure

```
sih-rag/
├── backend/                    # FastAPI Backend
│   ├── api/                    # API layer
│   │   ├── routes.py           # REST endpoints
│   │   └── schemas.py          # Pydantic request/response models
│   ├── core/                   # Business logic
│   │   ├── ollama_client.py    # Ollama HTTP client
│   │   ├── rag_engine.py       # RAG orchestration & prompts
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── observability.py    # Logging, metrics, tracing
│   ├── database/               # Data layer
│   │   ├── connection.py       # SQLAlchemy async setup
│   │   ├── models.py           # Document & Chunk ORM models
│   │   └── operations.py       # CRUD & vector search
│   ├── ingestion/              # Document processing
│   │   ├── pipeline.py         # Main ingestion orchestrator
│   │   ├── chunker.py          # Docling HybridChunker
│   │   └── embedder.py         # Ollama embedding generation
│   ├── config.py               # Centralized configuration
│   └── main.py                 # FastAPI application entry
│
├── frontend/                   # Next.js Frontend
│   ├── src/
│   │   ├── app/                # App Router pages
│   │   │   ├── page.tsx        # Home page with chat interface
│   │   │   └── globals.css     # Global styles & CSS variables
│   │   ├── components/         # React components
│   │   │   ├── ui/             # shadcn/ui primitives
│   │   │   └── chat-interface.tsx  # Main chat component
│   │   └── lib/                # Utilities & shared logic
│   │       ├── api.ts          # Backend API client
│   │       ├── store.ts        # Zustand state management
│   │       └── types.ts        # TypeScript interfaces
│   ├── public/                 # Static assets
│   └── package.json            # Dependencies
│
├── documents/                  # Place documents here for ingestion
├── sql/
│   └── schema.sql              # PostgreSQL schema with pgvector
├── docker-compose.yml          # Container orchestration
├── docker-compose.gpu.yml      # GPU-enabled variant
├── Dockerfile                  # Backend container
└── pyproject.toml              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended for easy deployment)
- **Ollama** installed and running locally
- **Node.js 18+** (for frontend)
- **FFmpeg** (optional, for audio transcription)

### Option 1: Full Stack with Docker (Recommended)

#### 1. Install Ollama

Download from [ollama.ai](https://ollama.ai)

```bash
# Pull required models
ollama pull mistral          # LLM for chat (7B parameters)
ollama pull nomic-embed-text # 768-dim embeddings

# Verify installation
ollama list
```

#### 2. Configure Environment

**Backend (.env in root):**
```bash
cp .env.example .env
```

Edit `.env`:
```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/offrag
OLLAMA_BASE_URL=http://host.docker.internal:11434  # Mac/Windows
# OLLAMA_BASE_URL=http://172.17.0.1:11434           # Linux
OLLAMA_LLM_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

#### 3. Start All Services

```bash
# Start all services (PostgreSQL + Backend + Frontend)
docker compose up -d

# Check logs
docker compose logs -f

# Check status
docker compose ps
```

**Access Points:**
- 🌐 **Frontend**: http://localhost:3000
- 🔌 **Backend API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

#### 4. Upload Documents

**Option A: Via Frontend (Easiest)**

1. Open http://localhost:3000
2. Click the paperclip icon (📎) in the chat input
3. Select a document (PDF, DOCX, PPTX, XLSX, MP3, etc.)
4. Wait for processing to complete
5. Start asking questions!

**Option B: Via Backend API**

```bash
# Using curl
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"

# Or use Swagger UI at http://localhost:8000/docs
```

**Option C: Batch Ingestion**

Process all files in the `documents/` folder:

```bash
# Using API endpoint
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_path": "documents", "clean_existing": false}'

# Or use ingestion container
docker-compose --profile ingestion up ingestion
```

**Supported formats:**
- 📄 **Documents**: PDF, DOCX, PPTX, XLSX
- 📝 **Text**: Markdown, TXT, HTML
- 🎵 **Audio**: MP3, WAV, M4A, FLAC (auto-transcribed)

#### 5. Using the Chat Interface

Once documents are uploaded:

1. **Ask Questions**: Type naturally in the chat box
   - "What are the key financial results?"
   - "Summarize the main points"
   - "What did John say about the budget?" (from audio)

2. **View Citations**: Click on citation cards to see sources with page numbers

3. **Manage Conversations**: 
   - Create new chats with "New Chat" button
   - Rename conversations by clicking the edit icon
   - Delete old conversations with the trash icon

4. **View Documents**: Click "X Docs" button to see uploaded files

#### 6. Stop Services

```bash
# Stop all containers
docker compose down

# Stop frontend (Ctrl+C in terminal)
```
---


Use this for development. Requires Python 3.9+ and PostgreSQL 15.

#### 1. Start PostgreSQL

```bash
docker compose up -d postgres
```

#### 2. Setup Backend

```bash
# Install dependencies
pip install uv
uv pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with DATABASE_URL=postgresql://postgres:postgres@localhost:5433/offrag

# Start backend
python -m backend.main
```

#### 3. Setup Frontend

```bash
cd frontend
npm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
```

Frontend at http://localhost:3000, Backend at http://localhost:8000

---

## 🐳 Docker Services

- **postgres** - PostgreSQL 15 with PGVector extension (port 5433)
- **backend** - FastAPI application with hot-reload (port 8000)
- **frontend** - Next.js application (port 3000)
- **ingestion** - One-time batch document processing (profile: ingestion)

### Docker Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# Rebuild after code changes
docker compose up -d --build

# View logs
docker compose logs -f
docker compose logs -f backend
docker compose logs -f frontend

# Restart specific service
docker compose restart backend
docker compose restart frontend

# Run ingestion job
docker compose --profile ingestion up ingestion
```

## 🎨 Frontend Architecture

### Tech Stack
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix UI primitives)
- **State Management**: Zustand with persistence
- **API Client**: Fetch API with Server-Sent Events (SSE)
- **Markdown Rendering**: react-markdown with syntax highlighting
- **Icons**: Lucide React


## 🔌 API Endpoints

### Health Check

```bash
GET /health
```

Returns database status, Ollama status, and knowledge base stats.

### Upload File

```bash
POST /upload
Content-Type: multipart/form-data

file: <your-file>
```

Upload and process a single document (PDF, DOCX, PPTX, XLSX, MD, TXT, MP3, WAV).

### Batch Ingestion

```bash
POST /ingest
Content-Type: application/json

{
  "documents_path": "documents",
  "clean_existing": false
}
```

Process all documents in a folder. Useful for initial bulk import.

### Chat (Non-streaming)

```bash
POST /chat
Content-Type: application/json

{
  "message": "What are the main topics?",
  "conversation_history": [...]  # optional
}
```

### Chat (Streaming)

```bash
POST /chat/stream
Content-Type: application/json

{
  "message": "Explain the key concepts",
  "conversation_history": [...]  # optional
}
```

Returns Server-Sent Events (SSE) with streaming response.

### Search Knowledge Base

```bash
POST /search
Content-Type: application/json

{
  "query": "machine learning",
  "limit": 5
}
```

Returns relevant chunks with similarity scores.

### List Documents

```bash
GET /documents?limit=100&offset=0
```

## ⚙️ Configuration

All settings in `.env` or environment variables:

### Database
- `DATABASE_URL` - PostgreSQL connection string
- `DB_POOL_MIN_SIZE` - Minimum pool size (default: 5)
- `DB_POOL_MAX_SIZE` - Maximum pool size (default: 20)

### Ollama
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_LLM_MODEL` - Chat model (default: mistral)
- `OLLAMA_EMBEDDING_MODEL` - Embedding model (default: nomic-embed-text)
- `OLLAMA_TIMEOUT` - Request timeout in seconds (default: 300)

### RAG
- `EMBEDDING_DIMENSIONS` - Vector dimensions (default: 768)
- `TOP_K_RESULTS` - Number of chunks to retrieve (default: 5)
- `SIMILARITY_THRESHOLD` - Minimum similarity score (default: 0.7)
- `MAX_TOKENS_PER_CHUNK` - Max tokens per chunk (default: 512)

### API
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: INFO)

### Frontend
- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://localhost:8000)



### Data Flow - Document Ingestion

```
User uploads file.pdf
       │
       ▼
┌─────────────────────────────────┐
│  Frontend                       │
│  - File upload via paperclip    │
│  - Progress tracking            │
└──────────────┬──────────────────┘
               │ POST /upload (multipart)
               ▼
┌─────────────────────────────────┐
│  Backend API                    │
│  - Validate file                │
│  - Save temporarily             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Ingestion Pipeline             │
│  1. Read (Docling converts)     │
│  2. Chunk (HybridChunker)       │
│  3. Embed (Ollama)              │
│  4. Store (PostgreSQL/PGVector) │
└──────────────┬────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  PostgreSQL + PGVector          │
│  - documents (metadata)         │
│  - chunks (text + vector(768))  │
└─────────────────────────────────┘
               │
               ▼
       Success response
               │
               ▼
┌─────────────────────────────────┐
│  Frontend                       │
│  - Show success toast           │
│  - Display chunk count          │
│  - Enable chat input            │
└─────────────────────────────────┘
```

### Data Flow - Chat Query

```
User types: "What are the financial results?"
       │
       ▼
┌─────────────────────────────────┐
│  Frontend                       │
│  - Add to conversation state    │
│  - Display user message         │
└──────────────┬──────────────────┘
               │ POST /chat/stream
               ▼
┌─────────────────────────────────┐
│  Backend API (SSE endpoint)     │
│  - Receive query                │
│  - Call RAG Engine              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  RAG Engine                     │
│  1. Generate query embedding    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Database Operations            │
│  - Vector similarity search     │
│  - Cosine distance < threshold  │
│  - Return top K chunks          │
└──────────────┬──────────────────┘
               │ Top 5 chunks + metadata
               ▼
┌─────────────────────────────────┐
│  RAG Engine                     │
│  2. Build prompt with context   │
│  3. Send citations via SSE      │
│  4. Call Ollama LLM (streaming) │
└──────────────┬──────────────────┘
               │ Stream tokens
               ▼
┌─────────────────────────────────┐
│  Backend API                    │
│  - Send SSE: citations          │
│  - Send SSE: tokens (streaming) │
│  - Send SSE: done signal        │
└──────────────┬──────────────────┘
               │ Server-Sent Events
               ▼
┌─────────────────────────────────┐
│  Frontend                       │
│  - Display citations first      │
│  - Stream assistant response    │
│  - Render markdown + code       │
│  - Save to conversation         │
└─────────────────────────────────┘
```

### Components Interaction

**1. Ollama Client** (`backend/core/ollama_client.py`)
   - HTTP client for Ollama API
   - Handles embeddings and chat completions
   - Streaming support for real-time responses

**2. RAG Engine** (`backend/core/rag_engine.py`)
   - Always-search pattern (retrieves context for every query)
   - Combines retrieval + generation in single flow
   - Prompt engineering with CRITICAL INSTRUCTIONS for focused answers
   - Citation tracking from search results

**3. Database Layer** (`backend/database/`)
   - SQLAlchemy async ORM models
   - PGVector integration for semantic search
   - CRUD operations with connection pooling
   - Cosine similarity search with configurable threshold

**4. Ingestion Pipeline** (`backend/ingestion/`)
   - Multi-format document processing via Docling
   - HybridChunker for semantic-aware splitting
   - Page number extraction from PDF metadata
   - Batch embedding generation
   - Audio transcription with Whisper ASR

**5. Frontend Components**
   - **ChatInterface**: Main conversation UI with streaming
   - **API Client**: SSE handling and error management
   - **Store**: Zustand for conversation persistence
   - **UI Components**: shadcn/ui with Radix primitives

## 🧪 Development

```

## 📊 Database Schema

### Documents Table

| Column     | Type      | Description              |
|------------|-----------|--------------------------|
| id         | UUID      | Primary key              |
| title      | TEXT      | Document title           |
| source     | TEXT      | Source path              |
| content    | TEXT      | Full document content    |
| metadata   | JSONB     | Additional metadata      |
| created_at | TIMESTAMP | Creation timestamp       |
| updated_at | TIMESTAMP | Last update timestamp    |

### Chunks Table

| Column       | Type       | Description                |
|--------------|------------|----------------------------|
| id           | UUID       | Primary key                |
| document_id  | UUID       | Foreign key to documents   |
| content      | TEXT       | Chunk text content         |
| embedding    | vector(768)| Embedding vector           |
| chunk_index  | INTEGER    | Chunk position in document |
| metadata     | JSONB      | Chunk metadata             |
| token_count  | INTEGER    | Number of tokens           |
| created_at   | TIMESTAMP  | Creation timestamp         |

## 🐛 Troubleshooting

### Frontend Issues

**Port 3000 already in use:**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
npm run dev -- -p 3001
```

**API connection errors:**
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check NEXT_PUBLIC_API_URL in frontend/.env.local
# Should match backend URL (http://localhost:8000)
```

**Build errors:**
```bash
# Clear Next.js cache
cd frontend
rm -rf .next node_modules
npm install
npm run dev
```

### Backend Issues

**Ollama Not Responding:**
```bash
# Check Ollama is running
ollama list

# Restart Ollama
ollama serve

# Test model
ollama run mistral "Hello"
```

**Database Connection Error:**
```bash
# Check PostgreSQL is running
docker compose ps

# Check logs
docker compose logs postgres

# Recreate database
docker compose down -v
docker compose up -d postgres
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install uv
uv pip install -e . --force-reinstall
```

**Slow embedding generation:**
- Use GPU-enabled Ollama for faster processing
- Reduce chunk size in ingestion settings
- Consider upgrading to more powerful Ollama models

**Reranker issues:**
- See [RERANKER.md](RERANKER.md) for detailed troubleshooting
- Disable with `RERANKER_ENABLED=false` if causing problems
- Check logs: `docker logs rag_backend | grep -i rerank`

---

## 📊 Monitoring & Observability

### Prometheus Metrics

Access raw metrics at: **http://localhost:8000/metrics**

Available metrics:
- `rag_requests_total` - Total chat requests
- `rag_search_latency_seconds` - Hybrid search performance
- `rag_generation_latency_seconds` - LLM response time
- `rag_chunks_retrieved_count` - Number of chunks per query
- `reranker_latency_seconds` - Reranker performance
- `reranker_rank_change_positions` - Ranking improvement
- `ingestion_*` - Document upload metrics
- `http_*` - Standard HTTP metrics

### Visual Dashboard

Access metrics dashboard at: **http://localhost:3000/metrics**

Features:
- Real-time performance graphs
- Request throughput monitoring
- Latency percentiles
- Auto-refresh every 10 seconds

### Structured Logging

All logs are JSON-formatted with:
- `timestamp` - ISO 8601 format
- `request_id` - Unique request tracking
- `level` - ERROR/WARNING/INFO/DEBUG
- `message` - Human-readable description
- `source` - File and line number

View logs:
```bash
docker compose logs -f backend | jq '.'
```

### Reranking Metrics

The reranker adds three specialized metrics:
- `reranker_latency_seconds` - Cross-encoder inference time
- `reranker_calls_total{status}` - Success/error counts
- `reranker_rank_change_positions` - Average position change after reranking

Higher rank change indicates the reranker is improving relevance.
---

**Built with ❤️ using:**
- **Frontend**: Next.js, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Database**: PostgreSQL with PGVector extension
- **LLM**: Ollama (Mistral for chat, nomic-embed-text for embeddings)
- **Document Processing**: Docling with Whisper ASR
