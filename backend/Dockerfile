# Dockerfile for RAG Backend - Works on both CPU and GPU systems
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    gcc \
    postgresql-client \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install UV package manager
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY backend ./backend
COPY documents ./documents

# Install Python dependencies
# PyTorch will auto-detect GPU/CPU at runtime
RUN uv pip install --system -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the backend
CMD ["python", "-m", "backend.main"]