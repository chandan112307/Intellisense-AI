# app/api/routes/feedback_router.py
"""
Feedback API endpoints.

POST /feedback/submit             — Submit user feedback on a response.
GET  /feedback/stats              — Aggregate feedback counts (optionally per user).
GET  /feedback/retrieval-boosts   — Chunk-level boost scores derived from feedback.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.core.logging import log_error, log_info
from app.feedback.feedback_store import (
    get_feedback_stats,
    get_retrieval_boost_from_feedback,
    store_feedback,
)
from app.intelligence.unified_learning import get_unified_learning

router = APIRouter(prefix="/feedback", tags=["feedback"])


# ─── Request / Response Schemas ───────────────────────────────────────────────


class FeedbackSubmitRequest(BaseModel):
    user_id: str
    query: str
    response: str
    feedback_type: str = Field(
        ...,
        description="One of: thumbs_up, thumbs_down, correction, edit",
    )
    feedback_text: Optional[str] = None
    chunk_ids: Optional[List[str]] = None
    confidence_score: Optional[float] = 0.0


class FeedbackSubmitResponse(BaseModel):
    status: str
    feedback_id: int


class FeedbackStatsResponse(BaseModel):
    total: int
    by_type: Dict[str, int]


class RetrievalBoostsResponse(BaseModel):
    boosts: Dict[str, float]


# ─── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/submit", response_model=FeedbackSubmitResponse)
async def submit_feedback(payload: FeedbackSubmitRequest):
    """Submit user feedback for a query/response pair."""
    try:
        # Store in legacy feedback_store for backward compatibility
        feedback_id = store_feedback(
            user_id=payload.user_id,
            query=payload.query,
            response=payload.response,
            feedback_type=payload.feedback_type,
            chunk_ids=payload.chunk_ids,
            feedback_text=payload.feedback_text,
            confidence_score=payload.confidence_score or 0.0,
        )
        # Also store in Unified Learning System for integrated improvement
        try:
            learning = get_unified_learning()
            learning.store_feedback(
                user_id=payload.user_id,
                query=payload.query,
                response=payload.response,
                feedback_type=payload.feedback_type,
                chunk_ids=payload.chunk_ids,
                feedback_text=payload.feedback_text or "",
                confidence_score=payload.confidence_score or 0.0,
            )
        except Exception as ule:
            log_info(f"Unified learning feedback skipped: {ule}")

        log_info(f"Feedback submitted: id={feedback_id} user={payload.user_id}")
        return FeedbackSubmitResponse(status="ok", feedback_id=feedback_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_error(f"Feedback submit failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store feedback")


@router.get("/stats", response_model=FeedbackStatsResponse)
async def feedback_stats(user_id: Optional[str] = Query(default=None)):
    """Return aggregate feedback counts, optionally scoped to a user."""
    try:
        stats = get_feedback_stats(user_id=user_id)
        return FeedbackStatsResponse(**stats)
    except Exception as e:
        log_error(f"Feedback stats failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback stats")


@router.get("/retrieval-boosts", response_model=RetrievalBoostsResponse)
async def retrieval_boosts():
    """Return chunk boost scores computed from historical feedback patterns."""
    try:
        boosts = get_retrieval_boost_from_feedback()
        return RetrievalBoostsResponse(boosts=boosts)
    except Exception as e:
        log_error(f"Retrieval boosts failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute retrieval boosts")
