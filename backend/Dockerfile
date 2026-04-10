# ─────────────────────────────────────────────────────────────────────────────
# RAG Backend Dockerfile
#
# Multi-arch: works on x86-64 (Lightning AI / cloud) AND arm64 (Mac M1/M2/M3)
# Python 3.11-slim · uv for fast dependency installs
# ─────────────────────────────────────────────────────────────────────────────

# TARGETPLATFORM is injected by BuildKit (linux/amd64 or linux/arm64).
# Using python:3.11-slim which ships both amd64 and arm64 layers.
FROM --platform=${TARGETPLATFORM:-linux/amd64} python:3.11-slim

# ── System dependencies ─────────────────────────────────────────────────────
# ffmpeg  : audio transcription (whisper)
# libpq-dev: asyncpg / psycopg2 native bindings
# curl    : healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install uv (fast Python package manager) ─────────────────────────────
RUN pip install --no-cache-dir uv

# ── Copy dependency manifests first (layer cache friendly) ────────────────
COPY pyproject.toml ./

# ── Copy source ───────────────────────────────────────────────────────────
COPY backend ./backend
COPY documents ./documents

# ── Install all Python deps ───────────────────────────────────────────────
# --system: install into the system Python (no venv needed inside container)
# --no-cache: keep image lean
RUN uv pip install --system --no-cache -e .

# ── Runtime env ──────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# ── Health check (generous start-period for model loading) ────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python", "-m", "backend.main"]