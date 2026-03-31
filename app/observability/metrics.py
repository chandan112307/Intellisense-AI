"""
Observability metrics module.

Tracks system metrics: latency, failure rate, grounded mode frequency,
token usage, and query volume. Uses in-memory tracking with thread-safe
access via threading.Lock.
"""

import threading
import time
from typing import Dict, List, Optional

from app.core.logging import log_info, log_error


class MetricsCollector:
    """Singleton metrics collector for system-wide observability."""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._data_lock = threading.Lock()
        self._reset_internal()
        log_info("MetricsCollector initialized")

    def _reset_internal(self) -> None:
        """Reset all internal counters (caller must hold _data_lock or be in __init__)."""
        self.query_count: int = 0
        self.total_latency_ms: int = 0
        self.failed_queries: int = 0
        self.grounded_mode_count: int = 0
        self.token_usage_total: int = 0
        self.queries_by_type: Dict[str, int] = {}
        self.latency_percentiles: List[int] = []
        self._endpoint_latencies: Dict[str, List[int]] = {}
        self._start_time: float = time.time()

    def record_query(
        self,
        latency_ms: int,
        success: bool,
        grounded_mode: bool,
        query_type: str,
        tokens_used: int = 0,
        endpoint: Optional[str] = None,
    ) -> None:
        """Record a single query's metrics."""
        with self._data_lock:
            self.query_count += 1
            self.total_latency_ms += latency_ms
            self.latency_percentiles.append(latency_ms)

            if not success:
                self.failed_queries += 1

            if grounded_mode:
                self.grounded_mode_count += 1

            self.token_usage_total += tokens_used
            self.queries_by_type[query_type] = (
                self.queries_by_type.get(query_type, 0) + 1
            )

            if endpoint:
                self._endpoint_latencies.setdefault(endpoint, []).append(latency_ms)

    def get_metrics_summary(self) -> dict:
        """Return a snapshot of all collected metrics."""
        with self._data_lock:
            count = self.query_count
            avg_latency = (
                round(self.total_latency_ms / count, 2) if count > 0 else 0.0
            )
            failure_rate = (
                round(self.failed_queries / count, 4) if count > 0 else 0.0
            )
            grounded_mode_rate = (
                round(self.grounded_mode_count / count, 4) if count > 0 else 0.0
            )
            avg_tokens = (
                round(self.token_usage_total / count, 2) if count > 0 else 0.0
            )

            p50, p95, p99 = self._compute_percentiles()

            endpoint_avg: Dict[str, float] = {}
            for ep, lats in self._endpoint_latencies.items():
                endpoint_avg[ep] = round(sum(lats) / len(lats), 2) if lats else 0.0

            uptime_seconds = round(time.time() - self._start_time, 2)

            return {
                "query_count": count,
                "total_latency_ms": self.total_latency_ms,
                "avg_latency_ms": avg_latency,
                "failure_rate": failure_rate,
                "failed_queries": self.failed_queries,
                "grounded_mode_count": self.grounded_mode_count,
                "grounded_mode_rate": grounded_mode_rate,
                "token_usage_total": self.token_usage_total,
                "avg_tokens_per_query": avg_tokens,
                "queries_by_type": dict(self.queries_by_type),
                "latency_p50": p50,
                "latency_p95": p95,
                "latency_p99": p99,
                "endpoint_avg_latency_ms": endpoint_avg,
                "uptime_seconds": uptime_seconds,
            }

    def get_health_status(self) -> dict:
        """Return system health indicators based on current metrics."""
        with self._data_lock:
            count = self.query_count
            failure_rate = (
                round(self.failed_queries / count, 4) if count > 0 else 0.0
            )
            avg_latency = (
                round(self.total_latency_ms / count, 2) if count > 0 else 0.0
            )

            if failure_rate >= 0.25:
                status = "unhealthy"
            elif failure_rate >= 0.10:
                status = "degraded"
            else:
                status = "healthy"

            return {
                "status": status,
                "query_count": count,
                "failure_rate": failure_rate,
                "avg_latency_ms": avg_latency,
                "failed_queries": self.failed_queries,
                "uptime_seconds": round(time.time() - self._start_time, 2),
            }

    def get_adaptive_signals(self) -> dict:
        """Return signals that can drive system self-optimization.

        Returns recommendations based on observed metrics patterns:
        - model_upgrade_needed: True if failure rate is high
        - expand_retrieval: True if grounded mode triggers frequently
        - reduce_latency: True if p95 latency exceeds threshold
        """
        with self._data_lock:
            count = self.query_count
            if count < 5:
                return {"sufficient_data": False}

            failure_rate = self.failed_queries / count
            grounded_rate = self.grounded_mode_count / count
            p50, p95, p99 = self._compute_percentiles()

            signals = {
                "sufficient_data": True,
                "failure_rate": failure_rate,
                "grounded_rate": grounded_rate,
                "model_upgrade_needed": failure_rate > 0.20,
                "expand_retrieval": grounded_rate > 0.30,
                "reduce_latency": p95 > 8000,
                "high_token_usage": (self.token_usage_total / count) > 800,
                "p95_latency_ms": p95,
            }
            return signals

    def reset_metrics(self) -> None:
        """Reset all metrics. Intended for testing."""
        with self._data_lock:
            self._reset_internal()
        log_info("MetricsCollector metrics reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_percentiles(self) -> tuple:
        """Compute p50, p95, p99 from recorded latencies. Caller must hold lock."""
        samples = sorted(self.latency_percentiles)
        if not samples:
            return (0, 0, 0)

        def _pct(pct: float) -> int:
            idx = min(int(len(samples) * pct + 0.5), len(samples) - 1)
            return samples[idx]

        return (_pct(0.50), _pct(0.95), _pct(0.99))


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

_collector = MetricsCollector()


def record_query(
    latency_ms: int,
    success: bool,
    grounded_mode: bool,
    query_type: str,
    tokens_used: int = 0,
    endpoint: Optional[str] = None,
) -> None:
    """Record a query's metrics via the singleton collector."""
    _collector.record_query(
        latency_ms=latency_ms,
        success=success,
        grounded_mode=grounded_mode,
        query_type=query_type,
        tokens_used=tokens_used,
        endpoint=endpoint,
    )


def get_metrics_summary() -> dict:
    """Return a metrics summary from the singleton collector."""
    return _collector.get_metrics_summary()


def get_health_status() -> dict:
    """Return system health status from the singleton collector."""
    return _collector.get_health_status()


def get_adaptive_signals() -> dict:
    """Return adaptive optimization signals from the singleton collector."""
    return _collector.get_adaptive_signals()


def reset_metrics() -> None:
    """Reset all metrics on the singleton collector."""
    _collector.reset_metrics()
