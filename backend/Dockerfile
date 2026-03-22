# ─────────────────────────────────────────────────────────────────────────────
# RAG Backend Dockerfile
#
# Optimised for Lightning AI (CPU + GPU compatible)
# Python 3.11-slim · uv for fast dependency installs
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install uv (fast Python package manager) ──────────────────────────────
RUN pip install --no-cache-dir uv

# ── Copy dependency manifests first (layer cache friendly) ────────────────
# Copy dependency manifests
COPY pyproject.toml ./

# Copy source BEFORE install
COPY backend ./backend
COPY documents ./documents

# Now install
RUN uv pip install --system --no-cache -e .

# ── Runtime env ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# ── Health check (generous start-period for model loading) ─────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python", "-m", "backend.main"]