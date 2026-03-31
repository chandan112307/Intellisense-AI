"""
Metrics API router.

Exposes endpoints for querying observability metrics and system health.
"""

from fastapi import APIRouter, HTTPException

from app.core.logging import log_info, log_error
from app.observability.metrics import (
    get_health_status,
    get_metrics_summary,
    reset_metrics,
)

router = APIRouter(prefix="/metrics", tags=["observability"])


@router.get("/summary")
async def metrics_summary():
    """Return the current metrics summary."""
    try:
        summary = get_metrics_summary()
        log_info("Metrics summary requested")
        return summary
    except Exception as e:
        log_error(f"Failed to retrieve metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics summary")


@router.get("/health")
async def metrics_health():
    """Return system health status based on observed metrics."""
    try:
        health = get_health_status()
        log_info(f"Health status requested: {health.get('status')}")
        return health
    except Exception as e:
        log_error(f"Failed to retrieve health status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve health status")


@router.post("/reset")
async def metrics_reset():
    """Reset all collected metrics."""
    try:
        reset_metrics()
        log_info("Metrics reset via API")
        return {"status": "ok", "message": "Metrics have been reset"}
    except Exception as e:
        log_error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset metrics")
