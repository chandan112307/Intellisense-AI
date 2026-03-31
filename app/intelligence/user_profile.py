"""
SQLite-backed user profile management for the intelligence layer.
Tracks knowledge levels, learning styles, and interaction history
to drive adaptive retrieval and response generation.
"""

import os
import json
import sqlite3
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.core.logging import log_info, log_warning, log_error

USER_PROFILE_DB_PATH = "data/user_profiles.db"

VALID_KNOWLEDGE_LEVELS = ("beginner", "intermediate", "advanced")
VALID_LEARNING_STYLES = ("visual", "textual", "interactive")

_ADAPTIVE_PARAMS = {
    "beginner": {"top_k": 3, "complexity": "simple", "add_explanations": True},
    "intermediate": {"top_k": 5, "complexity": "moderate", "add_explanations": False},
    "advanced": {"top_k": 8, "complexity": "detailed", "add_explanations": False},
}


class UserProfileStore:
    """Thread-safe singleton for user profile persistence."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = None):
        if self._initialized:
            return
        self.db_path = db_path or USER_PROFILE_DB_PATH
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_schema()
        self._initialized = True
        log_info(f"UserProfileStore initialized at {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    knowledge_level TEXT NOT NULL DEFAULT 'beginner',
                    weak_topics TEXT NOT NULL DEFAULT '[]',
                    learning_style TEXT NOT NULL DEFAULT 'textual',
                    query_count INTEGER NOT NULL DEFAULT 0,
                    correct_answers INTEGER NOT NULL DEFAULT 0,
                    total_answers INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS interaction_log (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query_text TEXT,
                    topic TEXT,
                    was_correct INTEGER,
                    difficulty TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_interaction_user
                    ON interaction_log(user_id, timestamp);
            """)
            conn.commit()
        finally:
            conn.close()

    # ── Profile CRUD ──

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Return the profile for a user, creating a default if absent."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                return self._row_to_dict(row)
            return self._create_default_profile(user_id, conn)
        finally:
            conn.close()

    def _create_default_profile(self, user_id: str, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Insert a default beginner profile and return it."""
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT INTO user_profiles
               (user_id, knowledge_level, weak_topics, learning_style,
                query_count, correct_answers, total_answers, created_at, updated_at)
               VALUES (?, 'beginner', '[]', 'textual', 0, 0, 0, ?, ?)""",
            (user_id, now, now),
        )
        conn.commit()
        log_info(f"Created default profile for user {user_id}")
        return {
            "user_id": user_id,
            "knowledge_level": "beginner",
            "weak_topics": [],
            "learning_style": "textual",
            "query_count": 0,
            "correct_answers": 0,
            "total_answers": 0,
            "created_at": now,
            "updated_at": now,
        }

    # ── Knowledge Level ──

    def update_knowledge_level(self, user_id: str, level: str) -> Dict[str, Any]:
        """Set the knowledge level for a user. Returns updated profile."""
        if level not in VALID_KNOWLEDGE_LEVELS:
            log_warning(f"Invalid knowledge level '{level}' for user {user_id}")
            raise ValueError(f"knowledge_level must be one of {VALID_KNOWLEDGE_LEVELS}")

        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        try:
            self._ensure_profile(user_id, conn)
            conn.execute(
                "UPDATE user_profiles SET knowledge_level = ?, updated_at = ? WHERE user_id = ?",
                (level, now, user_id),
            )
            conn.commit()
            log_info(f"User {user_id} knowledge level → {level}")
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            return self._row_to_dict(row)
        finally:
            conn.close()

    def update_learning_style(self, user_id: str, style: str) -> Dict[str, Any]:
        """Set the learning style for a user. Returns updated profile."""
        if style not in VALID_LEARNING_STYLES:
            log_warning(f"Invalid learning style '{style}' for user {user_id}")
            raise ValueError(f"learning_style must be one of {VALID_LEARNING_STYLES}")

        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        try:
            self._ensure_profile(user_id, conn)
            conn.execute(
                "UPDATE user_profiles SET learning_style = ?, updated_at = ? WHERE user_id = ?",
                (style, now, user_id),
            )
            conn.commit()
            log_info(f"User {user_id} learning style → {style}")
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            return self._row_to_dict(row)
        finally:
            conn.close()

    def update_weak_topics(self, user_id: str, weak_topics: List[str]) -> Dict[str, Any]:
        """Replace the weak-topics list for a user. Returns updated profile."""
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        try:
            self._ensure_profile(user_id, conn)
            conn.execute(
                "UPDATE user_profiles SET weak_topics = ?, updated_at = ? WHERE user_id = ?",
                (json.dumps(weak_topics), now, user_id),
            )
            conn.commit()
            log_info(f"User {user_id} weak topics updated ({len(weak_topics)} topics)")
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            return self._row_to_dict(row)
        finally:
            conn.close()

    # ── Interaction Tracking ──

    def record_interaction(
        self,
        user_id: str,
        query_text: str = None,
        topic: str = None,
        was_correct: bool = None,
        difficulty: str = None,
    ) -> Dict[str, Any]:
        """Record a user interaction and update aggregate counters. Returns updated profile."""
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        try:
            self._ensure_profile(user_id, conn)

            conn.execute(
                """INSERT INTO interaction_log
                   (user_id, query_text, topic, was_correct, difficulty, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, query_text, topic,
                 int(was_correct) if was_correct is not None else None,
                 difficulty, now),
            )

            sets = ["query_count = query_count + 1", "updated_at = ?"]
            params: list = [now]
            if was_correct is not None:
                sets.append("total_answers = total_answers + 1")
                if was_correct:
                    sets.append("correct_answers = correct_answers + 1")

            params.append(user_id)
            conn.execute(
                f"UPDATE user_profiles SET {', '.join(sets)} WHERE user_id = ?",
                tuple(params),
            )
            conn.commit()
            log_info(f"Recorded interaction for user {user_id}")

            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            return self._row_to_dict(row)
        except Exception as e:
            log_error(f"Failed to record interaction for {user_id}: {e}")
            raise
        finally:
            conn.close()

    # ── Adaptive Parameters ──

    def get_adaptive_params(self, user_id: str) -> Dict[str, Any]:
        """Return retrieval/response parameters tuned to the user's level.

        Keys returned:
            top_k          – number of retrieval results to fetch
            complexity      – response complexity tier (simple/moderate/detailed)
            add_explanations – whether to include step-by-step explanations
        """
        profile = self.get_user_profile(user_id)
        level = profile.get("knowledge_level", "beginner")
        params = dict(_ADAPTIVE_PARAMS.get(level, _ADAPTIVE_PARAMS["beginner"]))
        params["knowledge_level"] = level
        params["learning_style"] = profile.get("learning_style", "textual")
        params["weak_topics"] = profile.get("weak_topics", [])
        return params

    def adjust_difficulty(self, user_id: str) -> Dict[str, Any]:
        """Auto-adjust knowledge level based on answer accuracy.

        Rules:
            accuracy >= 80% and total_answers >= 5 → promote
            accuracy <  40% and total_answers >= 5 → demote
        Returns the (possibly updated) profile.
        """
        profile = self.get_user_profile(user_id)
        total = profile.get("total_answers", 0)
        if total < 5:
            return profile

        accuracy = profile.get("correct_answers", 0) / total
        current = profile["knowledge_level"]
        promotion_order = list(VALID_KNOWLEDGE_LEVELS)
        idx = promotion_order.index(current)

        if accuracy >= 0.8 and idx < len(promotion_order) - 1:
            new_level = promotion_order[idx + 1]
            log_info(f"Promoting user {user_id}: {current} → {new_level} (accuracy={accuracy:.0%})")
            return self.update_knowledge_level(user_id, new_level)

        if accuracy < 0.4 and idx > 0:
            new_level = promotion_order[idx - 1]
            log_info(f"Demoting user {user_id}: {current} → {new_level} (accuracy={accuracy:.0%})")
            return self.update_knowledge_level(user_id, new_level)

        return profile

    # ── Helpers ──

    def _ensure_profile(self, user_id: str, conn: sqlite3.Connection):
        """Create default profile if it doesn't exist yet."""
        row = conn.execute(
            "SELECT 1 FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            self._create_default_profile(user_id, conn)

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        """Convert sqlite3.Row to dict, parsing JSON fields."""
        if row is None:
            return {}
        d = dict(row)
        if "weak_topics" in d and isinstance(d["weak_topics"], str):
            try:
                d["weak_topics"] = json.loads(d["weak_topics"])
            except (json.JSONDecodeError, TypeError):
                d["weak_topics"] = []
        return d


# ── Module-level convenience functions ──

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Return profile dict for a user (creates default if needed)."""
    return UserProfileStore().get_user_profile(user_id)


def update_knowledge_level(user_id: str, level: str) -> Dict[str, Any]:
    """Set knowledge level for a user."""
    return UserProfileStore().update_knowledge_level(user_id, level)


def record_interaction(
    user_id: str,
    query_text: str = None,
    topic: str = None,
    was_correct: bool = None,
    difficulty: str = None,
) -> Dict[str, Any]:
    """Record an interaction and update counters."""
    return UserProfileStore().record_interaction(
        user_id, query_text=query_text, topic=topic,
        was_correct=was_correct, difficulty=difficulty,
    )


def get_adaptive_params(user_id: str) -> Dict[str, Any]:
    """Return adaptive retrieval/response params for a user."""
    return UserProfileStore().get_adaptive_params(user_id)


def adjust_difficulty(user_id: str) -> Dict[str, Any]:
    """Auto-adjust knowledge level based on answer accuracy."""
    return UserProfileStore().adjust_difficulty(user_id)
