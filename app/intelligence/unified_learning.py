# app/intelligence/unified_learning.py
"""
Unified Learning System.
Merges retrieval_memory, feedback_store, and data_flywheel into a single
learning layer that improves retrieval ranking, adapts confidence thresholds,
and learns from user feedback over time.
"""

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import log_info, log_error, log_warning

UNIFIED_LEARNING_DB_PATH = "data/unified_learning.db"


@dataclass
class LearningSignal:
    """A single learning signal from user interaction."""
    query: str
    query_type: str
    chunk_ids: List[str]
    confidence: float
    feedback_type: str  # thumbs_up, thumbs_down, correction, none
    feedback_text: str = ""
    response: str = ""
    outcome_quality: float = 0.0


class UnifiedLearningSystem:
    """
    Single learning layer that consolidates:
    - Retrieval memory (query type -> chunk type patterns)
    - Feedback store (user satisfaction signals)
    - Data flywheel (continuous improvement from usage)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        os.makedirs(os.path.dirname(UNIFIED_LEARNING_DB_PATH), exist_ok=True)
        self._conn = sqlite3.connect(UNIFIED_LEARNING_DB_PATH, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._db_lock = threading.Lock()
        self._decay_days = 30
        self._init_tables()
        log_info("UnifiedLearningSystem initialized")

    def _init_tables(self):
        with self._db_lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS retrieval_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    query_type TEXT DEFAULT '',
                    chunk_types TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 0.0,
                    recommendation TEXT DEFAULT 'proceed',
                    outcome_quality REAL DEFAULT 0.0,
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ro_query_type
                    ON retrieval_outcomes(query_type);

                CREATE TABLE IF NOT EXISTS feedback_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT DEFAULT '',
                    chunk_ids TEXT DEFAULT '[]',
                    feedback_type TEXT NOT NULL,
                    feedback_text TEXT DEFAULT '',
                    confidence_score REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fb_user
                    ON feedback_records(user_id);
                CREATE INDEX IF NOT EXISTS idx_fb_type
                    ON feedback_records(feedback_type);

                CREATE TABLE IF NOT EXISTS chunk_performance (
                    chunk_id TEXT PRIMARY KEY,
                    times_retrieved INTEGER DEFAULT 0,
                    times_positive INTEGER DEFAULT 0,
                    times_negative INTEGER DEFAULT 0,
                    avg_relevance REAL DEFAULT 0.0,
                    last_used REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS query_patterns (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_type TEXT DEFAULT '',
                    frequency INTEGER DEFAULT 1,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_feedback_score REAL DEFAULT 0.0,
                    last_seen REAL NOT NULL,
                    best_chunk_ids TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type TEXT DEFAULT '',
                    model_name TEXT NOT NULL,
                    reasoning_type TEXT DEFAULT 'single',
                    retrieval_strategy TEXT DEFAULT 'standard',
                    response_mode TEXT DEFAULT 'teaching',
                    confidence REAL DEFAULT 0.0,
                    success INTEGER DEFAULT 0,
                    feedback_score REAL DEFAULT 0.0,
                    latency_ms INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_do_query_type
                    ON decision_outcomes(query_type);
                CREATE INDEX IF NOT EXISTS idx_do_model
                    ON decision_outcomes(model_name);

                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    total_uses INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_feedback REAL DEFAULT 0.0,
                    avg_latency_ms REAL DEFAULT 0.0,
                    last_used REAL DEFAULT 0.0,
                    PRIMARY KEY (model_name, query_type)
                );

                CREATE TABLE IF NOT EXISTS failure_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type TEXT DEFAULT '',
                    failure_type TEXT NOT NULL,
                    model_name TEXT DEFAULT '',
                    retrieval_strategy TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    context_info TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fe_type
                    ON failure_events(failure_type);

                CREATE TABLE IF NOT EXISTS reasoning_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type TEXT DEFAULT '',
                    reasoning_type TEXT DEFAULT 'single',
                    steps_count INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.0,
                    success INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL
                );
            """)
            self._conn.commit()

    # ── Retrieval Memory ──

    def record_retrieval_outcome(
        self,
        query: str,
        query_type: str,
        chunk_types: List[str],
        confidence: float,
        recommendation: str,
        outcome_quality: float,
    ):
        """Record a retrieval outcome for pattern learning."""
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:32]
        now = time.time()
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO retrieval_outcomes
                   (query_hash, query_type, chunk_types, confidence, recommendation, outcome_quality, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (query_hash, query_type, json.dumps(chunk_types), confidence, recommendation, outcome_quality, now),
            )
            # Update query patterns
            row = self._conn.execute(
                "SELECT frequency, avg_confidence, avg_feedback_score FROM query_patterns WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
            if row:
                freq = row[0] + 1
                ac = (row[1] * row[0] + confidence) / freq
                self._conn.execute(
                    """UPDATE query_patterns SET frequency = ?, avg_confidence = ?,
                       last_seen = ? WHERE query_hash = ?""",
                    (freq, ac, now, query_hash),
                )
            else:
                self._conn.execute(
                    """INSERT INTO query_patterns
                       (query_hash, query_text, query_type, frequency, avg_confidence, last_seen)
                       VALUES (?, ?, ?, 1, ?, ?)""",
                    (query_hash, query, query_type, confidence, now),
                )
            self._conn.commit()

    def get_retrieval_boosts(self, query_type: str) -> Dict[str, float]:
        """Get chunk-type boost multipliers learned from retrieval history."""
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT chunk_types, confidence, outcome_quality, timestamp
                   FROM retrieval_outcomes WHERE query_type = ?""",
                (query_type,),
            ).fetchall()
        if not rows:
            return {}
        now = time.time()
        type_scores: Dict[str, List[float]] = {}
        for chunk_types_json, conf, quality, ts in rows:
            age_days = (now - ts) / 86400
            decay = math.exp(-age_days / self._decay_days)
            score = ((conf + quality) / 2) * decay
            for ct in json.loads(chunk_types_json):
                type_scores.setdefault(ct, []).append(score)

        boosts = {}
        for ct, scores in type_scores.items():
            avg = sum(scores) / len(scores)
            boosts[ct] = max(0.5, min(1.5, 0.75 + avg))
        return boosts

    def get_confidence_thresholds(self, query_type: str) -> Tuple[float, float]:
        """Get learned confidence thresholds from successful queries."""
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT confidence FROM retrieval_outcomes
                   WHERE query_type = ? AND outcome_quality > 0.5
                   ORDER BY timestamp DESC LIMIT 50""",
                (query_type,),
            ).fetchall()
        if not rows:
            return (0.70, 0.35)
        confidences = [r[0] for r in rows]
        avg_conf = sum(confidences) / len(confidences)
        high = min(0.85, max(0.50, avg_conf * 0.95))
        low = max(0.20, high * 0.50)
        return (high, low)

    # ── Feedback System ──

    def store_feedback(
        self,
        user_id: str,
        query: str,
        response: str,
        feedback_type: str,
        chunk_ids: Optional[List[str]] = None,
        feedback_text: str = "",
        confidence_score: float = 0.0,
    ) -> int:
        """Store user feedback and update chunk performance."""
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:32]
        chunk_ids = chunk_ids or []
        now = time.time()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        is_positive = feedback_type in ("thumbs_up", "correction")

        with self._db_lock:
            cursor = self._conn.execute(
                """INSERT INTO feedback_records
                   (user_id, query_hash, query, response, chunk_ids, feedback_type,
                    feedback_text, confidence_score, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, query_hash, query, response, json.dumps(chunk_ids),
                 feedback_type, feedback_text, confidence_score, ts),
            )
            feedback_id = cursor.lastrowid

            # Update chunk performance from feedback
            for cid in chunk_ids:
                row = self._conn.execute(
                    "SELECT times_retrieved, avg_relevance FROM chunk_performance WHERE chunk_id = ?",
                    (cid,),
                ).fetchone()
                if row:
                    tr = row[0] + 1
                    new_rel = (row[1] * row[0] + confidence_score) / tr
                    self._conn.execute(
                        """UPDATE chunk_performance SET times_retrieved = ?,
                           times_positive = times_positive + ?,
                           times_negative = times_negative + ?,
                           avg_relevance = ?, last_used = ?
                           WHERE chunk_id = ?""",
                        (tr, int(is_positive), int(not is_positive), new_rel, now, cid),
                    )
                else:
                    self._conn.execute(
                        """INSERT INTO chunk_performance
                           (chunk_id, times_retrieved, times_positive, times_negative,
                            avg_relevance, last_used)
                           VALUES (?, 1, ?, ?, ?, ?)""",
                        (cid, int(is_positive), int(not is_positive), confidence_score, now),
                    )

            # Update query pattern feedback score
            feedback_val = 1.0 if is_positive else -0.5
            row = self._conn.execute(
                "SELECT frequency, avg_feedback_score FROM query_patterns WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
            if row:
                freq = row[0]
                new_fb = (row[1] * freq + feedback_val) / (freq + 1)
                self._conn.execute(
                    "UPDATE query_patterns SET avg_feedback_score = ? WHERE query_hash = ?",
                    (new_fb, query_hash),
                )

            self._conn.commit()
        log_info(f"Stored feedback: type={feedback_type}, user={user_id}, chunks={len(chunk_ids)}")
        return feedback_id

    def get_feedback_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback statistics."""
        with self._db_lock:
            if user_id:
                rows = self._conn.execute(
                    "SELECT feedback_type, COUNT(*) FROM feedback_records WHERE user_id = ? GROUP BY feedback_type",
                    (user_id,),
                ).fetchall()
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM feedback_records WHERE user_id = ?", (user_id,),
                ).fetchone()[0]
            else:
                rows = self._conn.execute(
                    "SELECT feedback_type, COUNT(*) FROM feedback_records GROUP BY feedback_type",
                ).fetchall()
                total = self._conn.execute("SELECT COUNT(*) FROM feedback_records").fetchone()[0]
        return {"total": total, "by_type": {r[0]: r[1] for r in rows}}

    def get_chunk_boost(self, chunk_id: str) -> float:
        """Get a boost/penalty for a chunk based on feedback history."""
        with self._db_lock:
            row = self._conn.execute(
                "SELECT times_positive, times_negative, avg_relevance FROM chunk_performance WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if not row:
            return 0.0
        pos, neg, avg_rel = row
        total = pos + neg
        if total == 0:
            return 0.0
        feedback_ratio = (pos - neg) / total
        return round(feedback_ratio * 0.1 + avg_rel * 0.05, 4)

    def get_retrieval_boost_from_feedback(self) -> Dict[str, float]:
        """Get per-chunk boost multipliers from feedback patterns."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT chunk_id, times_positive, times_negative FROM chunk_performance"
            ).fetchall()
        boosts = {}
        for cid, pos, neg in rows:
            total = pos + neg
            if total >= 2:
                ratio = (pos - neg) / total
                boosts[cid] = max(0.5, min(1.5, 1.0 + ratio * 0.25))
        return boosts

    def get_negative_chunk_ids(self, threshold: int = 3) -> List[str]:
        """Get chunk IDs with consistent negative feedback."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM chunk_performance WHERE times_negative >= ?",
                (threshold,),
            ).fetchall()
        return [r[0] for r in rows]

    # ── Decision Learning ──

    def record_decision_outcome(
        self,
        query_type: str,
        model_name: str,
        reasoning_type: str,
        retrieval_strategy: str,
        response_mode: str,
        confidence: float,
        success: bool,
        latency_ms: int = 0,
    ):
        """Log a pipeline decision and its outcome for learning."""
        now = time.time()
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO decision_outcomes
                   (query_type, model_name, reasoning_type, retrieval_strategy,
                    response_mode, confidence, success, latency_ms, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (query_type, model_name, reasoning_type, retrieval_strategy,
                 response_mode, confidence, int(success), latency_ms, now),
            )
            # Update model performance aggregates
            row = self._conn.execute(
                "SELECT total_uses, successes, avg_confidence, avg_latency_ms FROM model_performance WHERE model_name = ? AND query_type = ?",
                (model_name, query_type),
            ).fetchone()
            if row:
                uses = row[0] + 1
                new_succ = row[1] + (1 if success else 0)
                new_conf = (row[2] * row[0] + confidence) / uses
                new_lat = (row[3] * row[0] + latency_ms) / uses
                self._conn.execute(
                    """UPDATE model_performance SET total_uses = ?, successes = ?,
                       avg_confidence = ?, avg_latency_ms = ?, last_used = ?
                       WHERE model_name = ? AND query_type = ?""",
                    (uses, new_succ, new_conf, new_lat, now, model_name, query_type),
                )
            else:
                self._conn.execute(
                    """INSERT INTO model_performance
                       (model_name, query_type, total_uses, successes, avg_confidence,
                        avg_latency_ms, last_used)
                       VALUES (?, ?, 1, ?, ?, ?, ?)""",
                    (model_name, query_type, int(success), confidence, float(latency_ms), now),
                )
            self._conn.commit()

    def get_best_model_for_query_type(self, query_type: str) -> Optional[str]:
        """Return the model with best weighted score for a query type.

        Uses weighted scoring (success_rate, confidence, recency) instead of
        a fixed threshold. Returns None if insufficient data.
        """
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT model_name, total_uses, successes, avg_confidence, last_used
                   FROM model_performance
                   WHERE query_type = ? AND total_uses >= 1""",
                (query_type,),
            ).fetchall()
        if not rows:
            return None
        now = time.time()
        best_model = None
        best_score = -1.0
        for model_name, total_uses, successes, avg_conf, last_used in rows:
            success_rate = successes / total_uses if total_uses > 0 else 0.0
            age_days = (now - last_used) / 86400 if last_used else self._decay_days
            recency = math.exp(-age_days / self._decay_days)
            # Data sufficiency factor: asymptotically approaches 1.0 as uses grow
            # (~0.39 at 1 use, ~0.63 at 2, ~0.78 at 3, ~0.95 at 6)
            sufficiency = 1.0 - math.exp(-total_uses / 2.0)
            score = (0.4 * success_rate + 0.3 * avg_conf + 0.3 * recency) * sufficiency
            if score > best_score:
                best_score = score
                best_model = model_name
        # Minimum viable recommendation threshold — below this, data is too weak
        if best_score < 0.15:
            return None
        return best_model

    def get_best_strategy_for_query_type(self, query_type: str) -> Optional[str]:
        """Return the retrieval strategy with best weighted score for a query type.

        Uses weighted scoring (success_rate, confidence, recency) instead of
        a fixed threshold.
        """
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT retrieval_strategy,
                       COUNT(*) as cnt,
                       AVG(confidence) as avg_conf,
                       SUM(success) as wins,
                       MAX(timestamp) as last_ts
                   FROM decision_outcomes
                   WHERE query_type = ? AND confidence > 0
                   GROUP BY retrieval_strategy
                   HAVING cnt >= 1""",
                (query_type,),
            ).fetchall()
        if not rows:
            return None
        now = time.time()
        best_strategy = None
        best_score = -1.0
        for strategy, cnt, avg_conf, wins, last_ts in rows:
            success_rate = wins / cnt if cnt > 0 else 0.0
            age_days = (now - last_ts) / 86400 if last_ts else self._decay_days
            recency = math.exp(-age_days / self._decay_days)
            sufficiency = 1.0 - math.exp(-cnt / 2.0)
            score = (0.4 * success_rate + 0.3 * avg_conf + 0.3 * recency) * sufficiency
            if score > best_score:
                best_score = score
                best_strategy = strategy
        if best_score < 0.15:
            return None
        return best_strategy

    def get_adaptive_top_k(self, query_type: str, base_top_k: int = 5) -> int:
        """Return an adjusted top_k based on recency-weighted historical success rates.

        Uses exponential decay to weight recent outcomes more heavily.
        If past retrievals for this query type had low confidence, increase top_k.
        If consistently high confidence, can reduce to save latency.
        """
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT confidence, timestamp
                   FROM retrieval_outcomes
                   WHERE query_type = ? AND timestamp > ?""",
                (query_type, time.time() - self._decay_days * 86400),
            ).fetchall()
        if not rows or len(rows) < 1:
            return base_top_k
        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0
        for conf, ts in rows:
            age_days = (now - ts) / 86400
            w = math.exp(-age_days / self._decay_days)
            weighted_sum += conf * w
            weight_total += w
        if weight_total == 0:
            return base_top_k
        avg_conf = weighted_sum / weight_total
        if avg_conf < 0.4:
            return min(base_top_k + 4, 15)  # Expand retrieval for low confidence
        if avg_conf > 0.8:
            return max(base_top_k - 1, 3)   # Can retrieve less for high confidence
        return base_top_k

    def get_confidence_trend(self, query_type: str, window: int = 10) -> Optional[float]:
        """Get the confidence trend for a query type from recent decision outcomes.

        Returns a value between -1.0 (declining) and 1.0 (improving).
        Returns None if insufficient data.
        """
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT confidence, timestamp
                   FROM decision_outcomes
                   WHERE query_type = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (query_type, window),
            ).fetchall()
        if len(rows) < 2:
            return None
        # Compare first half (recent) vs second half (older)
        mid = len(rows) // 2
        recent_avg = sum(r[0] for r in rows[:mid]) / mid
        older_avg = sum(r[0] for r in rows[mid:]) / (len(rows) - mid)
        # Trend: positive means improving, negative means declining
        return max(-1.0, min(1.0, recent_avg - older_avg))

    def get_all_models_for_query_type(self, query_type: str) -> List[str]:
        """Return all known model names for a given query type."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT DISTINCT model_name FROM model_performance WHERE query_type = ?",
                (query_type,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_all_strategies_for_query_type(self, query_type: str) -> List[str]:
        """Return all known retrieval strategies for a given query type."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT DISTINCT retrieval_strategy FROM decision_outcomes WHERE query_type = ?",
                (query_type,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_all_reasoning_types_for_query_type(self, query_type: str) -> List[str]:
        """Return all known reasoning types for a given query type."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT DISTINCT reasoning_type FROM reasoning_outcomes WHERE query_type = ?",
                (query_type,),
            ).fetchall()
        return [r[0] for r in rows]

    # ── Failure Learning ──

    VALID_FAILURE_TYPES = ("retrieval_failure", "reasoning_failure", "model_failure", "tool_failure")

    def classify_failure(
        self,
        confidence: float,
        has_chunks: bool,
        grounded_only: bool,
        reasoning_failed: bool,
        tool_failed: bool,
    ) -> str:
        """Classify a failure into one of the standard failure categories.

        Categories: retrieval_failure, reasoning_failure, model_failure, tool_failure.
        Always returns a value from VALID_FAILURE_TYPES.
        """
        if tool_failed:
            result = "tool_failure"
        elif not has_chunks or (confidence == 0.0 and not grounded_only):
            result = "retrieval_failure"
        elif reasoning_failed:
            result = "reasoning_failure"
        else:
            # Default: blame the model (low confidence with chunks present)
            result = "model_failure"
        assert result in self.VALID_FAILURE_TYPES
        return result

    def record_failure(
        self,
        query_type: str,
        failure_type: str,
        model_name: str = "",
        retrieval_strategy: str = "",
        confidence: float = 0.0,
        context_info: str = "",
    ):
        """Record a failure event for learning."""
        now = time.time()
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO failure_events
                   (query_type, failure_type, model_name, retrieval_strategy,
                    confidence, context_info, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (query_type, failure_type, model_name, retrieval_strategy,
                 confidence, context_info, now),
            )
            self._conn.commit()

    def get_failure_rate_by_type(self, query_type: str, window_days: int = 7) -> Dict[str, float]:
        """Get failure rates by failure_type for a query type within a time window."""
        cutoff = time.time() - window_days * 86400
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT failure_type, COUNT(*) FROM failure_events
                   WHERE query_type = ? AND timestamp > ?
                   GROUP BY failure_type""",
                (query_type, cutoff),
            ).fetchall()
            total = self._conn.execute(
                """SELECT COUNT(*) FROM decision_outcomes
                   WHERE query_type = ? AND timestamp > ?""",
                (query_type, cutoff),
            ).fetchone()[0]
        if total == 0:
            return {}
        return {ft: count / total for ft, count in rows}

    def should_upgrade_model(self, query_type: str) -> bool:
        """Check if failures suggest upgrading to a better model."""
        failure_rates = self.get_failure_rate_by_type(query_type)
        total_failure = sum(failure_rates.values())
        return total_failure > 0.3  # >30% failure rate triggers upgrade

    # ── Reasoning Outcomes ──

    def record_reasoning_outcome(
        self,
        query_type: str,
        reasoning_type: str,
        steps_count: int,
        confidence: float,
        success: bool,
    ):
        """Record a reasoning outcome for learning."""
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO reasoning_outcomes
                   (query_type, reasoning_type, steps_count, confidence, success, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (query_type, reasoning_type, steps_count, confidence, int(success), time.time()),
            )
            self._conn.commit()

    def get_best_reasoning_type(self, query_type: str) -> Optional[str]:
        """Return the reasoning type with best weighted score for a query type.

        Uses weighted scoring (success_rate, confidence, recency) instead of
        a fixed threshold.
        """
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT reasoning_type,
                       COUNT(*) as cnt,
                       SUM(success) as wins,
                       AVG(confidence) as avg_conf,
                       MAX(timestamp) as last_ts
                   FROM reasoning_outcomes
                   WHERE query_type = ?
                   GROUP BY reasoning_type
                   HAVING cnt >= 1""",
                (query_type,),
            ).fetchall()
        if not rows:
            return None
        now = time.time()
        best_type = None
        best_score = -1.0
        for reasoning_type, cnt, wins, avg_conf, last_ts in rows:
            success_rate = wins / cnt if cnt > 0 else 0.0
            age_days = (now - last_ts) / 86400 if last_ts else self._decay_days
            recency = math.exp(-age_days / self._decay_days)
            sufficiency = 1.0 - math.exp(-cnt / 2.0)
            score = (0.4 * success_rate + 0.3 * avg_conf + 0.3 * recency) * sufficiency
            if score > best_score:
                best_score = score
                best_type = reasoning_type
        if best_score < 0.15:
            return None
        return best_type

    # ── Flywheel Stats ──

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get overall learning system statistics."""
        with self._db_lock:
            ro_count = self._conn.execute("SELECT COUNT(*) FROM retrieval_outcomes").fetchone()[0]
            fb_count = self._conn.execute("SELECT COUNT(*) FROM feedback_records").fetchone()[0]
            cp_count = self._conn.execute("SELECT COUNT(*) FROM chunk_performance").fetchone()[0]
            qp_count = self._conn.execute("SELECT COUNT(*) FROM query_patterns").fetchone()[0]
            do_count = self._conn.execute("SELECT COUNT(*) FROM decision_outcomes").fetchone()[0]
            mp_count = self._conn.execute("SELECT COUNT(*) FROM model_performance").fetchone()[0]
            fe_count = self._conn.execute("SELECT COUNT(*) FROM failure_events").fetchone()[0]
            reo_count = self._conn.execute("SELECT COUNT(*) FROM reasoning_outcomes").fetchone()[0]
        return {
            "retrieval_outcomes_recorded": ro_count,
            "feedback_records": fb_count,
            "tracked_chunks": cp_count,
            "unique_query_patterns": qp_count,
            "decision_outcomes_logged": do_count,
            "model_performance_entries": mp_count,
            "failure_events": fe_count,
            "reasoning_outcomes": reo_count,
        }

    def cleanup_old(self, days: Optional[int] = None):
        """Remove old entries beyond decay window."""
        cutoff = time.time() - (days or self._decay_days * 3) * 86400
        with self._db_lock:
            self._conn.execute("DELETE FROM retrieval_outcomes WHERE timestamp < ?", (cutoff,))
            self._conn.commit()
        log_info(f"Cleaned up learning data older than {days or self._decay_days * 3} days")


def get_unified_learning() -> UnifiedLearningSystem:
    """Get the singleton UnifiedLearningSystem instance."""
    return UnifiedLearningSystem()
