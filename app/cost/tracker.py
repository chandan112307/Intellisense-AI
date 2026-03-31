"""
Cost & Model Optimization system for IntelliSense AI.

Tracks token usage, estimates cost per query, and implements
dynamic model selection based on query complexity and quality needs.
"""

import threading
import time
from collections import deque
from typing import Dict, Optional

from app.core.logging import log_info, log_error, log_warning

# ── Model Pricing (USD per 1K tokens) ──
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "llama-3.1-8b-instant": {
        "input": 0.00005,
        "output": 0.00008,
    },
    "llama-3.1-70b-versatile": {
        "input": 0.00059,
        "output": 0.00079,
    },
    "mixtral-8x7b-32768": {
        "input": 0.00024,
        "output": 0.00024,
    },
    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006,
    },
    "gpt-4o": {
        "input": 0.0025,
        "output": 0.01,
    },
    "claude-3-5-sonnet": {
        "input": 0.003,
        "output": 0.015,
    },
}

# ── Complexity / Quality → Model mapping tiers ──
_TIER_CHEAPEST = "llama-3.1-8b-instant"
_TIER_BALANCED = "llama-3.1-70b-versatile"
_TIER_BEST = "claude-3-5-sonnet"

_COST_HISTORY_LIMIT = 100


class CostTracker:
    """Singleton tracker for token usage and cost across all models."""

    _instance: Optional["CostTracker"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._data_lock = threading.Lock()
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.queries_tracked: int = 0
        self.cost_by_model: Dict[str, float] = {}
        self.cost_history: deque = deque(maxlen=_COST_HISTORY_LIMIT)
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "CostTracker":
        """Return the singleton CostTracker instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton — primarily for testing."""
        with cls._lock:
            cls._instance = None


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts.

    Returns 0.0 and logs a warning when the model is unknown.
    """
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        log_warning(f"Unknown model for cost estimation: {model_name}")
        return 0.0

    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost


def record_usage(
    model_name: str, input_tokens: int, output_tokens: int
) -> Dict[str, object]:
    """Record token usage and return a cost breakdown dict.

    Returns:
        {
            "model": str,
            "input_tokens": int,
            "output_tokens": int,
            "input_cost_usd": float,
            "output_cost_usd": float,
            "total_cost_usd": float,
            "timestamp": float,
        }
    """
    tracker = CostTracker.get_instance()
    pricing = MODEL_PRICING.get(model_name)

    if pricing is None:
        log_warning(f"Unknown model for usage recording: {model_name}")
        input_cost = 0.0
        output_cost = 0.0
    else:
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

    total = input_cost + output_cost
    ts = time.time()

    entry = {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total,
        "timestamp": ts,
    }

    with tracker._data_lock:
        tracker.total_input_tokens += input_tokens
        tracker.total_output_tokens += output_tokens
        tracker.total_cost_usd += total
        tracker.queries_tracked += 1
        tracker.cost_by_model[model_name] = (
            tracker.cost_by_model.get(model_name, 0.0) + total
        )
        tracker.cost_history.append(entry)

    log_info(
        f"Cost recorded: model={model_name} "
        f"tokens={input_tokens}+{output_tokens} cost=${total:.6f}"
    )
    return entry


def get_cost_summary() -> Dict[str, object]:
    """Return an aggregate cost summary.

    Returns:
        {
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_cost_usd": float,
            "queries_tracked": int,
            "average_cost_per_query": float,
            "cost_by_model": dict,
        }
    """
    tracker = CostTracker.get_instance()

    with tracker._data_lock:
        avg = (
            tracker.total_cost_usd / tracker.queries_tracked
            if tracker.queries_tracked > 0
            else 0.0
        )
        return {
            "total_input_tokens": tracker.total_input_tokens,
            "total_output_tokens": tracker.total_output_tokens,
            "total_cost_usd": tracker.total_cost_usd,
            "queries_tracked": tracker.queries_tracked,
            "average_cost_per_query": avg,
            "cost_by_model": dict(tracker.cost_by_model),
        }


def select_model(query_complexity: str, quality_required: str) -> str:
    """Select the most cost-effective model for the given requirements.

    Args:
        query_complexity: "simple", "moderate", or "complex".
        quality_required: "low", "medium", "high", or "critical".

    Returns:
        Model name string from MODEL_PRICING.
    """
    complexity = query_complexity.lower()
    quality = quality_required.lower()

    # Critical / verification — always use best model
    if quality in ("critical", "verification"):
        selected = _TIER_BEST
    elif complexity == "simple" and quality == "low":
        selected = _TIER_CHEAPEST
    elif complexity in ("complex",) or quality == "high":
        selected = _TIER_BALANCED
    else:
        # moderate complexity or medium quality
        selected = _TIER_CHEAPEST

    log_info(
        f"Model selected: {selected} "
        f"(complexity={complexity}, quality={quality})"
    )
    return selected


def get_budget_status(daily_budget_usd: float = 10.0) -> Dict[str, object]:
    """Return current budget status with remaining budget and projections.

    Args:
        daily_budget_usd: Maximum daily spend in USD (default $10).

    Returns:
        {
            "daily_budget_usd": float,
            "total_spent_usd": float,
            "remaining_budget_usd": float,
            "queries_tracked": int,
            "average_cost_per_query": float,
            "projected_queries_remaining": int | None,
            "budget_utilization_pct": float,
        }
    """
    tracker = CostTracker.get_instance()

    with tracker._data_lock:
        spent = tracker.total_cost_usd
        remaining = max(daily_budget_usd - spent, 0.0)
        queries = tracker.queries_tracked
        avg = spent / queries if queries > 0 else 0.0
        projected = int(remaining / avg) if avg > 0 else None
        utilization = (spent / daily_budget_usd * 100) if daily_budget_usd > 0 else 0.0

    if utilization >= 90:
        log_warning(
            f"Budget nearly exhausted: {utilization:.1f}% used "
            f"(${spent:.4f} / ${daily_budget_usd:.2f})"
        )

    return {
        "daily_budget_usd": daily_budget_usd,
        "total_spent_usd": spent,
        "remaining_budget_usd": remaining,
        "queries_tracked": queries,
        "average_cost_per_query": avg,
        "projected_queries_remaining": projected,
        "budget_utilization_pct": utilization,
    }
