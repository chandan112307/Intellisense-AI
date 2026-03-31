# app/flywheel/data_flywheel.py
"""
Data Flywheel: Continuously learns from queries, feedback, and usage patterns
to improve retrieval ranking and prompt quality over time.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from app.core.logging import log_info, log_error

FLYWHEEL_DB_PATH = "data/flywheel.db"


class DataFlywheel:
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
        os.makedirs(os.path.dirname(FLYWHEEL_DB_PATH), exist_ok=True)
        self._conn = sqlite3.connect(FLYWHEEL_DB_PATH, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._db_lock = threading.Lock()
        self._init_tables()
        log_info("DataFlywheel initialized")

    def _init_tables(self):
        with self._db_lock:
            self._conn.executescript("""
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
                CREATE TABLE IF NOT EXISTS chunk_performance (
                    chunk_id TEXT PRIMARY KEY,
                    times_retrieved INTEGER DEFAULT 0,
                    times_positive_feedback INTEGER DEFAULT 0,
                    times_negative_feedback INTEGER DEFAULT 0,
                    avg_relevance_score REAL DEFAULT 0.0,
                    last_used REAL DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS prompt_effectiveness (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_template TEXT NOT NULL,
                    times_used INTEGER DEFAULT 1,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_feedback REAL DEFAULT 0.0,
                    last_used REAL NOT NULL
                );
            """)
            self._conn.commit()

    def record_query_outcome(
        self,
        query: str,
        query_type: str,
        confidence: float,
        feedback_score: float,
        chunk_ids: List[str],
    ):
        """Record a query outcome for pattern learning."""
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:32]
        now = time.time()
        chunk_ids_json = json.dumps(chunk_ids[:10])
        with self._db_lock:
            row = self._conn.execute(
                "SELECT frequency, avg_confidence, avg_feedback_score FROM query_patterns WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
            if row:
                freq, ac, af = row
                new_freq = freq + 1
                new_ac = (ac * freq + confidence) / new_freq
                new_af = (af * freq + feedback_score) / new_freq
                self._conn.execute(
                    """UPDATE query_patterns SET frequency = ?, avg_confidence = ?,
                       avg_feedback_score = ?, last_seen = ?, best_chunk_ids = ?
                       WHERE query_hash = ?""",
                    (new_freq, new_ac, new_af, now, chunk_ids_json, query_hash),
                )
            else:
                self._conn.execute(
                    """INSERT INTO query_patterns
                       (query_hash, query_text, query_type, frequency, avg_confidence,
                        avg_feedback_score, last_seen, best_chunk_ids)
                       VALUES (?, ?, ?, 1, ?, ?, ?, ?)""",
                    (query_hash, query, query_type, confidence, feedback_score, now, chunk_ids_json),
                )
            self._conn.commit()

    def record_chunk_feedback(
        self, chunk_id: str, relevance_score: float, positive: bool
    ):
        """Record chunk retrieval outcome for ranking improvement."""
        now = time.time()
        with self._db_lock:
            row = self._conn.execute(
                "SELECT times_retrieved, avg_relevance_score FROM chunk_performance WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                tr, ars = row
                new_tr = tr + 1
                new_ars = (ars * tr + relevance_score) / new_tr
                self._conn.execute(
                    """UPDATE chunk_performance SET times_retrieved = ?,
                       times_positive_feedback = times_positive_feedback + ?,
                       times_negative_feedback = times_negative_feedback + ?,
                       avg_relevance_score = ?, last_used = ?
                       WHERE chunk_id = ?""",
                    (new_tr, int(positive), int(not positive), new_ars, now, chunk_id),
                )
            else:
                self._conn.execute(
                    """INSERT INTO chunk_performance
                       (chunk_id, times_retrieved, times_positive_feedback,
                        times_negative_feedback, avg_relevance_score, last_used)
                       VALUES (?, 1, ?, ?, ?, ?)""",
                    (chunk_id, int(positive), int(not positive), relevance_score, now),
                )
            self._conn.commit()

    def get_chunk_boost(self, chunk_id: str) -> float:
        """Get a boost/penalty score for a chunk based on historical performance."""
        with self._db_lock:
            row = self._conn.execute(
                """SELECT times_positive_feedback, times_negative_feedback, avg_relevance_score
                   FROM chunk_performance WHERE chunk_id = ?""",
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

    def get_popular_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most frequent query patterns."""
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT query_text, query_type, frequency, avg_confidence, avg_feedback_score
                   FROM query_patterns ORDER BY frequency DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "query": r[0], "query_type": r[1], "frequency": r[2],
                "avg_confidence": r[3], "avg_feedback": r[4],
            }
            for r in rows
        ]

    def get_flywheel_stats(self) -> Dict[str, Any]:
        """Get overall flywheel statistics."""
        with self._db_lock:
            qp_count = self._conn.execute("SELECT COUNT(*) FROM query_patterns").fetchone()[0]
            cp_count = self._conn.execute("SELECT COUNT(*) FROM chunk_performance").fetchone()[0]
            total_queries = self._conn.execute(
                "SELECT COALESCE(SUM(frequency), 0) FROM query_patterns"
            ).fetchone()[0]
        return {
            "unique_query_patterns": qp_count,
            "tracked_chunks": cp_count,
            "total_queries_recorded": total_queries,
        }


def get_data_flywheel() -> DataFlywheel:
    return DataFlywheel()
