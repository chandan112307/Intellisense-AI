# app/scalability/rate_limiter.py
"""
Scalability Layer: Rate limiting and concurrent user handling.
Provides middleware and utilities for request throttling.
"""

import time
import threading
from collections import defaultdict
from typing import Dict, Optional, Tuple

from app.core.logging import log_info, log_warning


class RateLimiter:
    """Token-bucket rate limiter with per-user and global limits."""

    def __init__(
        self,
        global_rpm: int = 200,
        per_user_rpm: int = 30,
        burst_allowance: int = 5,
    ):
        self._global_rpm = global_rpm
        self._per_user_rpm = per_user_rpm
        self._burst_allowance = burst_allowance
        self._lock = threading.Lock()
        self._global_requests: list = []
        self._user_requests: Dict[str, list] = defaultdict(list)

    def _cleanup_window(self, timestamps: list, window_seconds: float = 60.0) -> list:
        """Remove timestamps outside the sliding window."""
        cutoff = time.time() - window_seconds
        return [t for t in timestamps if t > cutoff]

    def check_rate_limit(self, user_id: Optional[str] = None) -> Tuple[bool, str]:
        """Check if a request is allowed. Returns (allowed, reason)."""
        now = time.time()
        with self._lock:
            self._global_requests = self._cleanup_window(self._global_requests)
            if len(self._global_requests) >= self._global_rpm:
                log_warning(f"Global rate limit exceeded: {len(self._global_requests)}/{self._global_rpm}")
                return False, "global_rate_limit_exceeded"

            if user_id:
                self._user_requests[user_id] = self._cleanup_window(
                    self._user_requests[user_id]
                )
                limit = self._per_user_rpm + self._burst_allowance
                if len(self._user_requests[user_id]) >= limit:
                    log_warning(f"User rate limit exceeded: user={user_id}")
                    return False, "user_rate_limit_exceeded"

            self._global_requests.append(now)
            if user_id:
                self._user_requests[user_id].append(now)

        return True, "ok"

    def get_usage_stats(self, user_id: Optional[str] = None) -> Dict:
        """Get current rate limit usage statistics."""
        with self._lock:
            self._global_requests = self._cleanup_window(self._global_requests)
            stats = {
                "global_requests_in_window": len(self._global_requests),
                "global_limit": self._global_rpm,
                "active_users": len(self._user_requests),
            }
            if user_id:
                self._user_requests[user_id] = self._cleanup_window(
                    self._user_requests[user_id]
                )
                stats["user_requests_in_window"] = len(self._user_requests[user_id])
                stats["user_limit"] = self._per_user_rpm + self._burst_allowance
            return stats

    def reset(self):
        """Reset all rate limit counters."""
        with self._lock:
            self._global_requests.clear()
            self._user_requests.clear()


_default_limiter = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    global _default_limiter
    with _limiter_lock:
        if _default_limiter is None:
            _default_limiter = RateLimiter()
        return _default_limiter
