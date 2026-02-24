# RAG AI Engineer Technical Test

VoiceFlip Technologies | February 2026

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- HuggingFace API token

### Setup (5 minutes)

```bash
# Clone and setup
git clone https://github.com/your-github/rag-project.git
cd rag-project
export HF_TOKEN="hf_your_token_here"

# Install dependencies
pip install -e ".[dev]"

# Start services
docker compose up -d
sleep 20
docker compose ps

# Test endpoints
curl http://localhost:8000/healthz
```

### Usage

**Ingest documents:**
```bash
python -m src.commands.ingest_cli --corpus-dir data/corpus
```

**Index corpus:**
```bash
python -m src.commands.index_cli --strategy recursive
```

**Query via CLI:**
```bash
python -m src.commands.query_cli "What is LangGraph?"
```

**Start API server:**
```bash
uvicorn src.main:app --port 8000
```

**Start Chainlit UI (separate terminal):**
```bash
chainlit run src/ui/chainlit_app.py --port 8501
```

Open browser: `http://localhost:8501`

### Services

- **API**: `http://localhost:8000` (FastAPI)
- **UI**: `http://localhost:8501` (Chainlit)
- **Vector DB**: `http://localhost:6333` (Qdrant)
- **Cache**: `http://localhost:6379` (Redis)

## 📋 Documentation

See [MASTER_ROADMAP.md](MASTER_ROADMAP.md) for complete specification.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_config.py -v
```

## 🛠️ Development

```bash
# Lint
ruff check . --fix

# Type check
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 📚 Stack

- **LLM**: Qwen/Qwen2.5-1.5B-Instruct (HF Inference API)
- **Embeddings**: all-MiniLM-L6-v2 (HF Inference API, 384 dim)
- **Vector DB**: Qdrant (Docker, persistent)
- **Agent**: LangGraph StateGraph (6 nodes, rule-based routing)
- **UI**: Chainlit (step visibility, separate process)
- **API**: FastAPI (6 endpoints, rate limiting)
- **Cache**: Redis (dedup, embedding cache, rate-limit, memory)
- **Eval**: RAGAS (faithfulness, relevancy, precision)

## 📝 Environment Variables

```bash
# Required
HF_TOKEN=hf_xxxxxxxxxxxxx

# Auto-configured in Docker
QDRANT_URL=http://vectordb:6333
REDIS_URL=redis://redis:6379

# Optional
TAVILY_API_KEY=tvly_xxxxx
LOG_LEVEL=INFO
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

## 📞 Support

See [docs/](docs/) for integration guides and troubleshooting.

---

**Status**: FINAL | ULTRA-COMPLETE | ZERO-CONTRADICTION | LOCKED
