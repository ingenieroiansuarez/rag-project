"""
FastAPI Server Entrypoint
"""
import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI
import requests
import redis.asyncio as redis

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG AI Engineer API",
    description="Technical Test API per mastering roadmap",
    version="0.1.0",
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/readyz")
async def readyz() -> dict[str, bool | dict[str, str]]:
    """
    Readiness probe validating external dependencies:
    - HF_TOKEN presence
    - Qdrant availability
    - Redis availability
    """
    results: dict[str, str] = {
        "hf_token": "ok" if settings.hf_token else "missing",
        "qdrant": "unknown",
        "redis": "unknown",
    }

    # Test Qdrant (timeout early to avoid hanging)
    try:
        # Use httpx or requests for sync probing since qdrant_client is heavier
        # The readiness should just check network/service availability quickly
        resp = await asyncio.to_thread(
            requests.get, f"{settings.qdrant_url}/healthz", timeout=2.0
        )
        if resp.status_code == 200:
            results["qdrant"] = "ok"
        else:
            results["qdrant"] = f"error_{resp.status_code}"
    except Exception as e:
        logger.error(f"Qdrant readiness probe failed: {e}")
        results["qdrant"] = "unreachable"

    # Test Redis
    try:
        redis_client = redis.from_url(settings.redis_url)
        if await redis_client.ping():
            results["redis"] = "ok"
        else:
            results["redis"] = "failed_ping"
        await redis_client.aclose()
    except Exception as e:
        logger.error(f"Redis readiness probe failed: {e}")
        results["redis"] = "unreachable"

    is_ready = all(v == "ok" for v in results.values())
    
    return {
        "ready": is_ready,
        "details": results,
    }
