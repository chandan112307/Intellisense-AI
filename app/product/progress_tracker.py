# app/product/progress_tracker.py
"""
Product Layer: Progress tracking, topic mastery, study history, revision suggestions.
SQLite-backed persistence for user learning progress.
"""

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.logging import log_info, log_error

PROGRESS_DB_PATH = "data/progress.db"


@dataclass
class TopicProgress:
    topic: str
    queries_count: int = 0
    correct_count: int = 0
    mastery_score: float = 0.0
    last_studied: float = 0.0
    needs_revision: bool = False


@dataclass
class StudySession:
    session_id: str
    user_id: str
    topic: str
    query: str
    timestamp: float
    confidence: float
    was_helpful: bool = True


class ProgressTracker:
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
        os.makedirs(os.path.dirname(PROGRESS_DB_PATH), exist_ok=True)
        self._conn = sqlite3.connect(PROGRESS_DB_PATH, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._db_lock = threading.Lock()
        self._init_tables()
        log_info("ProgressTracker initialized")

    def _init_tables(self):
        with self._db_lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS topic_progress (
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    queries_count INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    mastery_score REAL DEFAULT 0.0,
                    last_studied REAL DEFAULT 0.0,
                    PRIMARY KEY (user_id, topic)
                );
                CREATE TABLE IF NOT EXISTS study_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    query TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    was_helpful INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_study_user
                    ON study_history(user_id);
                CREATE INDEX IF NOT EXISTS idx_study_topic
                    ON study_history(user_id, topic);
            """)
            self._conn.commit()

    def record_study(
        self,
        user_id: str,
        session_id: str,
        topic: str,
        query: str,
        confidence: float,
        was_helpful: bool = True,
    ):
        """Record a study interaction and update topic progress."""
        now = time.time()
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO study_history
                   (session_id, user_id, topic, query, timestamp, confidence, was_helpful)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, user_id, topic, query, now, confidence, int(was_helpful)),
            )
            self._conn.execute(
                """INSERT INTO topic_progress (user_id, topic, queries_count, correct_count, mastery_score, last_studied)
                   VALUES (?, ?, 1, ?, ?, ?)
                   ON CONFLICT(user_id, topic) DO UPDATE SET
                       queries_count = queries_count + 1,
                       correct_count = correct_count + ?,
                       mastery_score = ?,
                       last_studied = ?""",
                (
                    user_id, topic, int(was_helpful), confidence, now,
                    int(was_helpful), confidence, now,
                ),
            )
            self._conn.commit()
        log_info(f"Recorded study: user={user_id} topic={topic}")

    def get_topic_progress(self, user_id: str) -> List[TopicProgress]:
        """Get progress for all topics for a user."""
        now = time.time()
        week_seconds = 7 * 24 * 3600
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT topic, queries_count, correct_count, mastery_score, last_studied
                   FROM topic_progress WHERE user_id = ? ORDER BY last_studied DESC""",
                (user_id,),
            ).fetchall()
        results = []
        for topic, qc, cc, ms, ls in rows:
            needs_revision = (now - ls) > week_seconds and ms < 0.8
            results.append(TopicProgress(
                topic=topic, queries_count=qc, correct_count=cc,
                mastery_score=ms, last_studied=ls, needs_revision=needs_revision,
            ))
        return results

    def get_study_history(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent study history for a user."""
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT session_id, topic, query, timestamp, confidence, was_helpful
                   FROM study_history WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "session_id": r[0], "topic": r[1], "query": r[2],
                "timestamp": r[3], "confidence": r[4], "was_helpful": bool(r[5]),
            }
            for r in rows
        ]

    def get_revision_suggestions(self, user_id: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Suggest topics that need revision based on time decay and mastery."""
        progress = self.get_topic_progress(user_id)
        candidates = [
            tp for tp in progress
            if tp.needs_revision or tp.mastery_score < 0.6
        ]
        candidates.sort(key=lambda tp: tp.mastery_score)
        suggestions = []
        for tp in candidates[:max_suggestions]:
            suggestions.append({
                "topic": tp.topic,
                "mastery_score": tp.mastery_score,
                "last_studied": tp.last_studied,
                "reason": "low_mastery" if tp.mastery_score < 0.6 else "time_decay",
            })
        return suggestions

    def get_overall_stats(self, user_id: str) -> Dict[str, Any]:
        """Get overall learning statistics for a user."""
        progress = self.get_topic_progress(user_id)
        if not progress:
            return {
                "topics_studied": 0, "total_queries": 0,
                "avg_mastery": 0.0, "topics_needing_revision": 0,
            }
        total_queries = sum(tp.queries_count for tp in progress)
        avg_mastery = sum(tp.mastery_score for tp in progress) / len(progress)
        revision_count = sum(1 for tp in progress if tp.needs_revision)
        return {
            "topics_studied": len(progress),
            "total_queries": total_queries,
            "avg_mastery": round(avg_mastery, 3),
            "topics_needing_revision": revision_count,
        }


def get_progress_tracker() -> ProgressTracker:
    return ProgressTracker()
