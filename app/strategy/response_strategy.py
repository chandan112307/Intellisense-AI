# app/strategy/response_strategy.py
"""
Response Strategy Layer.
Dynamically chooses response mode (teaching, exam, hint, summary)
based on user profile and query characteristics.
"""

from dataclasses import dataclass
from typing import Optional
from app.core.logging import log_info


@dataclass
class ResponseStrategy:
    mode: str  # teaching, exam, hint, summary, detailed
    tone: str  # encouraging, neutral, formal
    include_examples: bool
    include_references: bool
    max_length_hint: int  # suggested max tokens for response


STRATEGY_CONFIGS = {
    "teaching": ResponseStrategy(
        mode="teaching", tone="encouraging",
        include_examples=True, include_references=True, max_length_hint=600,
    ),
    "exam": ResponseStrategy(
        mode="exam", tone="neutral",
        include_examples=False, include_references=False, max_length_hint=200,
    ),
    "hint": ResponseStrategy(
        mode="hint", tone="encouraging",
        include_examples=False, include_references=False, max_length_hint=100,
    ),
    "summary": ResponseStrategy(
        mode="summary", tone="neutral",
        include_examples=False, include_references=True, max_length_hint=150,
    ),
    "detailed": ResponseStrategy(
        mode="detailed", tone="formal",
        include_examples=True, include_references=True, max_length_hint=1000,
    ),
}


def detect_strategy(
    query: str,
    knowledge_level: str = "intermediate",
    preferences: Optional[dict] = None,
) -> ResponseStrategy:
    """Select response strategy based on query and user profile."""
    query_lower = query.lower().strip()
    preferences = preferences or {}

    explicit_mode = preferences.get("response_mode")
    if explicit_mode and explicit_mode in STRATEGY_CONFIGS:
        log_info(f"Using explicit response mode: {explicit_mode}")
        return STRATEGY_CONFIGS[explicit_mode]

    if any(kw in query_lower for kw in ["quiz me", "test me", "examine", "assess"]):
        return STRATEGY_CONFIGS["exam"]

    if any(kw in query_lower for kw in ["hint", "clue", "nudge", "help me think"]):
        return STRATEGY_CONFIGS["hint"]

    if any(kw in query_lower for kw in [
        "summarize", "summary", "brief", "tldr", "in short", "overview",
    ]):
        return STRATEGY_CONFIGS["summary"]

    if any(kw in query_lower for kw in [
        "explain in detail", "elaborate", "deep dive", "comprehensive",
    ]):
        return STRATEGY_CONFIGS["detailed"]

    if knowledge_level == "beginner":
        return STRATEGY_CONFIGS["teaching"]

    return STRATEGY_CONFIGS["teaching"]


def build_strategy_prompt_prefix(strategy: ResponseStrategy) -> str:
    """Build a prompt prefix that instructs the LLM to follow the strategy."""
    parts = [f"Respond in {strategy.mode} mode with a {strategy.tone} tone."]

    if strategy.include_examples:
        parts.append("Include relevant examples where helpful.")
    else:
        parts.append("Do not include examples.")

    if strategy.include_references:
        parts.append("Cite sources when possible.")

    parts.append(
        f"Keep the response concise, aiming for roughly {strategy.max_length_hint} tokens."
    )

    return " ".join(parts)
