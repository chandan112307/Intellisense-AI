"""
Multi-Step Reasoning Engine.

Implements query decomposition, per-step validation against context,
and reasoning-chain construction for complex or compound queries.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.logging import log_info, log_warning

# ── Constants ────────────────────────────────────────────────────────────────

_COMPOUND_SPLITTERS = re.compile(
    r"\b(?:and also|and additionally|additionally|also|and)\b",
    re.IGNORECASE,
)

_COMPARISON_KEYWORDS = re.compile(
    r"\b(?:compare|contrast|difference|differences|differ|versus|vs\.?|"
    r"similarities|similarity|similar|distinguish|unlike|whereas)\b",
    re.IGNORECASE,
)

_NUMBERED_PARTS = re.compile(r"(?:^|\n)\s*(?:\d+[.)]\s|[-•]\s)")

_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "shall", "would",
        "should", "may", "might", "must", "can", "could", "of", "in", "to",
        "for", "with", "on", "at", "by", "from", "as", "into", "about",
        "between", "through", "after", "before", "during", "it", "its",
        "this", "that", "these", "those", "i", "we", "you", "he", "she",
        "they", "me", "us", "him", "her", "them", "my", "our", "your",
        "his", "their", "what", "which", "who", "whom", "how", "when",
        "where", "why", "if", "or", "but", "not", "no", "so", "than",
        "too", "very", "just", "only", "and",
    }
)

_VALIDATION_THRESHOLD = 0.15


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """A single step in a multi-step reasoning chain."""

    step_number: int
    sub_query: str
    context: str = ""
    answer: str = ""
    confidence: float = 0.0
    validated: bool = False


@dataclass
class ReasoningResult:
    """Aggregated result of the full reasoning process."""

    original_query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    reasoning_chain: str = ""
    confidence: float = 0.0
    is_multi_step: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Lowercase tokenisation with stop-word removal."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


def _compute_overlap(answer_tokens: set, context_tokens: set) -> float:
    """Return the fraction of answer tokens present in context tokens."""
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _extract_chunk_text(chunk) -> str:
    """Safely extract text from a context chunk."""
    return getattr(chunk, "text", "") or ""


def _gather_context_tokens(context_chunks: list) -> set:
    """Combine all chunk texts into a single token set."""
    combined = " ".join(_extract_chunk_text(c) for c in context_chunks)
    return _tokenize(combined)


def _best_context_for_query(query: str, context_chunks: list) -> str:
    """Select the chunk whose tokens overlap most with the sub-query."""
    query_tokens = _tokenize(query)
    if not query_tokens or not context_chunks:
        return ""

    best_text = ""
    best_score = -1.0
    for chunk in context_chunks:
        text = _extract_chunk_text(chunk)
        chunk_tokens = _tokenize(text)
        if not chunk_tokens:
            continue
        score = len(query_tokens & chunk_tokens) / len(query_tokens)
        if score > best_score:
            best_score = score
            best_text = text

    return best_text


# ── Public API ───────────────────────────────────────────────────────────────

def needs_multi_step(query: str) -> bool:
    """Determine whether *query* requires multi-step decomposition.

    Returns ``True`` when the query contains comparison keywords,
    multiple question marks, numbered/bulleted parts, or ``and``
    joining distinct topics.
    """
    if _COMPARISON_KEYWORDS.search(query):
        return True

    if query.count("?") > 1:
        return True

    if _NUMBERED_PARTS.search(query):
        return True

    # "and" joining distinct clauses (at least two verb-bearing clauses)
    parts = _COMPOUND_SPLITTERS.split(query)
    if len(parts) >= 2:
        meaningful = [p.strip() for p in parts if len(p.strip().split()) >= 3]
        if len(meaningful) >= 2:
            return True

    return False


def decompose_query(query: str) -> List[str]:
    """Break a complex query into a list of simpler sub-queries.

    Handles:
    * Compound questions joined by *and / also / additionally*.
    * Comparison questions — extracts both sides.
    * Multi-part questions with numbered items or bullet points.

    Returns the original query as a single-element list when no
    decomposition is applicable.
    """
    query = query.strip()
    if not query:
        return [query]

    # Numbered / bulleted lists
    items = _NUMBERED_PARTS.split(query)
    items = [s.strip() for s in items if s.strip()]
    if len(items) >= 2:
        log_info(f"Decomposed query into {len(items)} numbered parts")
        return items

    # Comparison questions
    if _COMPARISON_KEYWORDS.search(query):
        between_match = re.search(
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
            query,
            re.IGNORECASE,
        )
        if between_match:
            side_a = between_match.group(1).strip()
            side_b = between_match.group(2).strip()
            log_info("Decomposed comparison query into two sides")
            return [
                f"What is {side_a}?",
                f"What is {side_b}?",
                query,
            ]
        # Fallback: keep original as single query for simple comparisons
        return [query]

    # Compound splitters
    parts = _COMPOUND_SPLITTERS.split(query)
    parts = [p.strip().rstrip("?").strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        meaningful = [p for p in parts if len(p.split()) >= 3]
        if len(meaningful) >= 2:
            sub_queries = [p + "?" if not p.endswith("?") else p for p in meaningful]
            log_info(f"Decomposed compound query into {len(sub_queries)} sub-queries")
            return sub_queries

    return [query]


def validate_reasoning_step(
    step: ReasoningStep,
    context_chunks: list,
) -> bool:
    """Check whether *step.answer* is supported by the provided context.

    Validation passes when the token overlap between the answer and the
    combined context exceeds ``_VALIDATION_THRESHOLD`` (0.15).
    """
    if not step.answer:
        return False

    context_tokens = _gather_context_tokens(context_chunks)
    answer_tokens = _tokenize(step.answer)
    overlap = _compute_overlap(answer_tokens, context_tokens)

    supported = overlap > _VALIDATION_THRESHOLD
    step.validated = supported
    if not supported:
        log_warning(
            f"Step {step.step_number} failed validation "
            f"(overlap={overlap:.2f}, threshold={_VALIDATION_THRESHOLD})"
        )
    return supported


def build_reasoning_chain(steps: List[ReasoningStep]) -> str:
    """Combine validated step answers into a coherent reasoning chain."""
    if not steps:
        return ""

    parts: List[str] = []
    for step in steps:
        header = f"Step {step.step_number}: {step.sub_query}"
        body = step.answer if step.answer else "(no answer)"
        status = "validated" if step.validated else "unvalidated"
        parts.append(f"{header}\n{body} [{status}]")

    return "\n\n".join(parts)


def reason(
    query: str,
    context_chunks: list,
) -> ReasoningResult:
    """Execute the full reasoning pipeline for *query*.

    1. Decide whether multi-step decomposition is needed.
    2. Decompose into sub-queries (or keep as single step).
    3. For each sub-query, gather the best matching context and
       produce a preliminary answer from that context.
    4. Validate every step against the available context chunks.
    5. Build a reasoning chain and aggregate confidence.

    Each element of *context_chunks* is expected to expose a ``.text``
    attribute containing the chunk's textual content.
    """
    log_info(f"Reasoning engine invoked for query: {query[:120]}")
    multi_step = needs_multi_step(query)
    sub_queries = decompose_query(query) if multi_step else [query]

    steps: List[ReasoningStep] = []
    for idx, sub_q in enumerate(sub_queries, start=1):
        best_ctx = _best_context_for_query(sub_q, context_chunks)
        answer = best_ctx if best_ctx else ""

        step = ReasoningStep(
            step_number=idx,
            sub_query=sub_q,
            context=best_ctx,
            answer=answer,
            confidence=1.0 if best_ctx else 0.0,
        )
        validate_reasoning_step(step, context_chunks)
        steps.append(step)

    chain = build_reasoning_chain(steps)

    # Aggregate confidence: mean of validated steps (or all if none validated)
    validated_steps = [s for s in steps if s.validated]
    scoring_steps = validated_steps if validated_steps else steps
    avg_confidence = (
        sum(s.confidence for s in scoring_steps) / len(scoring_steps)
        if scoring_steps
        else 0.0
    )

    # Final answer: prefer validated answers
    answer_parts = [s.answer for s in (validated_steps or steps) if s.answer]
    final_answer = " ".join(answer_parts) if answer_parts else ""

    result = ReasoningResult(
        original_query=query,
        steps=steps,
        final_answer=final_answer,
        reasoning_chain=chain,
        confidence=avg_confidence,
        is_multi_step=multi_step,
    )
    log_info(
        f"Reasoning complete: {len(steps)} step(s), "
        f"multi_step={multi_step}, confidence={avg_confidence:.2f}"
    )
    return result
