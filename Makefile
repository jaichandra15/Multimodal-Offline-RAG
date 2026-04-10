# ═════════════════════════════════════════════════════════════════════════════
# Multimodal Offline RAG — Makefile
#
# Usage:
#   make setup      — copy .env.example → .env  (first time only)
#   make start      — start all services (CPU)
#   make start-gpu  — start all services (GPU / Lightning AI)
#   make start-git-gpu - pull latest code and start all services with GPU
#   make stop       — stop all services
#   make restart    — restart all services
#   make ingest     — ingest documents from ./documents folder
#   make logs       — tail logs from all services
#   make logs-back  — tail backend logs only
#   make status     — show running containers and health
#   make clean      — stop and remove containers + volumes (destructive!)
#   make rebuild    — rebuild all images from scratch
# ═════════════════════════════════════════════════════════════════════════════

.PHONY: setup start start-gpu start-m1 stop restart ingest logs logs-back status clean rebuild help

COMPOSE       = docker compose
COMPOSE_GPU   = docker compose -f docker-compose.yml -f docker-compose.gpu.yml
COMPOSE_M1    = docker compose -f docker-compose.yml -f docker-compose.m1.yml

# ── First-time setup ──────────────────────────────────────────────────────────
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅  .env created from .env.example"; \
		echo "👉  Edit .env and set NEXT_PUBLIC_API_URL to your Lightning AI URL"; \
	else \
		echo "ℹ️   .env already exists — skipping"; \
	fi

# ── Start / Stop ──────────────────────────────────────────────────────────────
start: _check_env
	@echo "🚀  Starting RAG stack (CPU)..."
	$(COMPOSE) up -d --build
	@echo ""
	@echo "✅  Services started:"
	@echo "    Frontend  → http://localhost:3000"
	@echo "    Backend   → http://localhost:8000"
	@echo "    Ollama    → http://localhost:11434"
	@echo ""
	@echo "    Run 'make logs' to follow logs"

start-git-gpu:_check_env
	@echo "🚀  Starting RAG stack (GPU)..."
	git pull
	$(COMPOSE_GPU) up -d --build
	@echo ""
	@echo "✅  Services started (GPU mode)"
	@echo "    Frontend  → http://localhost:3000"
	@echo "    Backend   → http://localhost:8000"

start-gpu: _check_env
	@echo "🚀  Starting RAG stack (GPU)..."
	$(COMPOSE_GPU) up -d --build
	@echo ""
	@echo "✅  Services started (GPU mode)"
	@echo "    Frontend  → http://localhost:3000"
	@echo "    Backend   → http://localhost:8000"

start-m1: _check_env
	@echo "🍎  Starting RAG stack (Mac M1/M2/M3 — Apple Silicon)..."
	@echo "    Platform : linux/arm64  •  LLM backend : Gemini / phi3:mini"
	DOCKER_DEFAULT_PLATFORM=linux/arm64 $(COMPOSE_M1) up -d --build
	@echo ""
	@echo "✅  Services started (M1 mode)"
	@echo "    Frontend  → http://localhost:3000"
	@echo "    Backend   → http://localhost:8000"
	@echo "    Ollama    → http://localhost:11434"
	@echo ""
	@echo "    Tip: first run pulls phi3:mini and nomic-embed-text (~2.5 GB total)"
	@echo "    Run 'make logs' to follow progress"

stop:
	@echo "🛑  Stopping all services..."
	$(COMPOSE) down
	@echo "✅  All services stopped"

restart: stop start

# ── Document Ingestion ────────────────────────────────────────────────────────
ingest: _check_env
	@echo "📚  Running document ingestion from ./documents ..."
	$(COMPOSE) --profile ingestion run --rm ingestion
	@echo "✅  Ingestion complete"

# ── Logs ──────────────────────────────────────────────────────────────────────
logs:
	$(COMPOSE) logs -f --tail=100

logs-back:
	$(COMPOSE) logs -f --tail=100 backend

logs-front:
	$(COMPOSE) logs -f --tail=100 frontend

# ── Status ────────────────────────────────────────────────────────────────────
status:
	@echo "📊  Container status:"
	$(COMPOSE) ps
	@echo ""
	@echo "🩺  Backend health:"
	@curl -sf http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "   (backend not reachable)"

# ── Clean / Rebuild ───────────────────────────────────────────────────────────
clean:
	@echo "⚠️   This will delete ALL containers AND volumes (database data)."
	@read -p "    Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(COMPOSE) down -v --remove-orphans
	@echo "✅  Clean complete"

rebuild:
	@echo "🔨  Rebuilding all images (no cache)..."
	$(COMPOSE) build --no-cache
	@echo "✅  Rebuild complete — run 'make start' to launch"

# ── Internal helpers ──────────────────────────────────────────────────────────
_check_env:
	@if [ ! -f .env ]; then \
		echo "❌  No .env file found. Run 'make setup' first."; \
		exit 1; \
	fi

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Multimodal Offline RAG — Available Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup       Create .env from template (run once)"
	@echo "  make start       Start all services (CPU / x86)"
	@echo "  make start-m1    Start all services (Mac M1/M2/M3 Apple Silicon)"
	@echo "  make start-gpu   Start all services (GPU / Lightning AI)"
	@echo "  make stop        Stop all services"
	@echo "  make restart     Restart all services"
	@echo "  make ingest      Ingest docs from ./documents folder"
	@echo "  make logs        Follow all service logs"
	@echo "  make logs-back   Follow backend logs only"
	@echo "  make status      Show container status + backend health"
	@echo "  make rebuild     Rebuild images from scratch (no cache)"
	@echo "  make clean       ⚠️  Remove containers AND volumes"
	@echo ""
