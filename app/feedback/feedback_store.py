# app/feedback/feedback_store.py
"""
Feedback Loop System.

SQLite-backed store that captures user feedback (thumbs up/down, edits,
corrections) alongside the original query, response, and retrieved chunk IDs.
Provides retrieval-boost signals so future queries can re-rank chunks based
on historical feedback patterns.

Table: feedback
    id               INTEGER  — auto-increment primary key
    user_id          TEXT     — who submitted the feedback
    query            TEXT     — original user query
    response         TEXT     — system response that was evaluated
    chunk_ids        TEXT     — JSON list of chunk IDs used in the response
    feedback_type    TEXT     — thumbs_up | thumbs_down | correction | edit
    feedback_text    TEXT     — optional free-form comment / correction text
    confidence_score REAL     — model confidence at time of response
    timestamp        TEXT     — ISO-8601 creation timestamp
"""

import json
import os
import sqlite3
import threading
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.logging import log_error, log_info, log_warning

# ─── Constants ────────────────────────────────────────────────────────────────

FEEDBACK_DB_PATH = "data/feedback.db"

_POSITIVE_TYPES = {"thumbs_up"}
_NEGATIVE_TYPES = {"thumbs_down"}
_NEGATIVE_THRESHOLD = 3  # min negative hits before a chunk is flagged

_DB_INIT_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    response         TEXT    NOT NULL,
    chunk_ids        TEXT    NOT NULL DEFAULT '[]',
    feedback_type    TEXT    NOT NULL,
    feedback_text    TEXT    DEFAULT NULL,
    confidence_score REAL    NOT NULL DEFAULT 0.0,
    timestamp        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_user     ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type     ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
"""

# ─── Thread-safe connection helper ────────────────────────────────────────────

_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """Return a new SQLite connection with WAL mode and row-factory enabled."""
    os.makedirs(os.path.dirname(FEEDBACK_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_schema() -> None:
    """Create the feedback table if it does not already exist."""
    conn = _get_conn()
    try:
        conn.executescript(_DB_INIT_SQL)
        conn.commit()
    finally:
        conn.close()


# Initialise schema on module import
_ensure_schema()


# ─── Public API ───────────────────────────────────────────────────────────────


def store_feedback(
    user_id: str,
    query: str,
    response: str,
    feedback_type: str,
    chunk_ids: Optional[List[str]] = None,
    feedback_text: Optional[str] = None,
    confidence_score: float = 0.0,
) -> int:
    """Persist a single feedback record. Returns the new row id."""
    allowed_types = {"thumbs_up", "thumbs_down", "correction", "edit"}
    if feedback_type not in allowed_types:
        raise ValueError(f"feedback_type must be one of {allowed_types}")

    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        with _lock:
            cur = conn.execute(
                """INSERT INTO feedback
                   (user_id, query, response, chunk_ids, feedback_type,
                    feedback_text, confidence_score, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    query,
                    response,
                    json.dumps(chunk_ids or []),
                    feedback_type,
                    feedback_text,
                    confidence_score,
                    now,
                ),
            )
            conn.commit()
            row_id = cur.lastrowid
        log_info(f"Stored feedback id={row_id} type={feedback_type} user={user_id}")
        return row_id
    except Exception as e:
        log_error(f"Failed to store feedback: {e}")
        raise
    finally:
        conn.close()


def get_feedback_for_query(query: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all feedback rows matching *query* (exact match), optionally filtered by user."""
    conn = _get_conn()
    try:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE query = ? AND user_id = ? ORDER BY timestamp DESC",
                (query, user_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE query = ? ORDER BY timestamp DESC",
                (query,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        log_error(f"Failed to get feedback for query: {e}")
        return []
    finally:
        conn.close()


def get_feedback_stats(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate feedback counts grouped by type, optionally scoped to a user."""
    conn = _get_conn()
    try:
        if user_id:
            rows = conn.execute(
                "SELECT feedback_type, COUNT(*) as cnt FROM feedback WHERE user_id = ? GROUP BY feedback_type",
                (user_id,),
            ).fetchall()
            total_row = conn.execute(
                "SELECT COUNT(*) as total FROM feedback WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        else:
            rows = conn.execute(
                "SELECT feedback_type, COUNT(*) as cnt FROM feedback GROUP BY feedback_type",
            ).fetchall()
            total_row = conn.execute("SELECT COUNT(*) as total FROM feedback").fetchone()

        by_type = {r["feedback_type"]: r["cnt"] for r in rows}
        total = total_row["total"] if total_row else 0
        return {"total": total, "by_type": by_type}
    except Exception as e:
        log_error(f"Failed to get feedback stats: {e}")
        return {"total": 0, "by_type": {}}
    finally:
        conn.close()


def get_retrieval_boost_from_feedback() -> Dict[str, float]:
    """Compute chunk_id → boost_score from cumulative feedback.

    Positive feedback (thumbs_up) adds +1 per occurrence; negative feedback
    (thumbs_down) subtracts 1.  The raw tally is normalised to a boost
    centred around 1.0 (range 0.5 – 1.5).
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT chunk_ids, feedback_type FROM feedback"
        ).fetchall()

        scores: Counter = Counter()
        counts: Counter = Counter()

        for row in rows:
            chunk_ids = json.loads(row["chunk_ids"])
            ftype = row["feedback_type"]
            delta = 1.0 if ftype in _POSITIVE_TYPES else (-1.0 if ftype in _NEGATIVE_TYPES else 0.0)
            for cid in chunk_ids:
                scores[cid] += delta
                counts[cid] += 1

        if not scores:
            return {}

        max_abs = max(abs(v) for v in scores.values()) or 1.0
        boosts: Dict[str, float] = {}
        for cid, raw in scores.items():
            normalised = raw / max_abs  # range [-1, 1]
            boosts[cid] = round(1.0 + 0.5 * normalised, 3)  # range [0.5, 1.5]

        log_info(f"Computed retrieval boosts for {len(boosts)} chunks")
        return boosts
    except Exception as e:
        log_error(f"Failed to compute retrieval boosts: {e}")
        return {}
    finally:
        conn.close()


def get_negative_chunk_ids() -> List[str]:
    """Return chunk IDs that consistently receive negative feedback.

    A chunk is flagged when its total negative feedback count meets or
    exceeds ``_NEGATIVE_THRESHOLD``.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT chunk_ids FROM feedback WHERE feedback_type IN ('thumbs_down')"
        ).fetchall()

        neg_counts: Counter = Counter()
        for row in rows:
            for cid in json.loads(row["chunk_ids"]):
                neg_counts[cid] += 1

        flagged = [cid for cid, cnt in neg_counts.items() if cnt >= _NEGATIVE_THRESHOLD]
        if flagged:
            log_warning(f"{len(flagged)} chunk(s) flagged with consistent negative feedback")
        return flagged
    except Exception as e:
        log_error(f"Failed to get negative chunk ids: {e}")
        return []
    finally:
        conn.close()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, deserialising JSON fields."""
    d = dict(row)
    if "chunk_ids" in d and isinstance(d["chunk_ids"], str):
        try:
            d["chunk_ids"] = json.loads(d["chunk_ids"])
        except (json.JSONDecodeError, TypeError):
            d["chunk_ids"] = []
    return d
