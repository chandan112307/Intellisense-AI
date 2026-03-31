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

    # ── Flywheel Stats ──

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get overall learning system statistics."""
        with self._db_lock:
            ro_count = self._conn.execute("SELECT COUNT(*) FROM retrieval_outcomes").fetchone()[0]
            fb_count = self._conn.execute("SELECT COUNT(*) FROM feedback_records").fetchone()[0]
            cp_count = self._conn.execute("SELECT COUNT(*) FROM chunk_performance").fetchone()[0]
            qp_count = self._conn.execute("SELECT COUNT(*) FROM query_patterns").fetchone()[0]
        return {
            "retrieval_outcomes_recorded": ro_count,
            "feedback_records": fb_count,
            "tracked_chunks": cp_count,
            "unique_query_patterns": qp_count,
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
