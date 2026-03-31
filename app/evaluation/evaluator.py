"""
Automated Evaluation Pipeline for the RAG system.

Measures retrieval recall, answer correctness, hallucination rate,
and confidence calibration to assess end-to-end system quality.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.logging import log_info, log_error, log_warning


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvaluationCase:
    """A single evaluation test case."""

    query: str
    expected_answer: str
    expected_chunk_ids: Optional[List[str]] = None
    category: str = "factual"       # factual | conceptual | edge_case
    difficulty: str = "medium"      # easy | medium | hard


@dataclass
class EvaluationResult:
    """Result produced by evaluating one case."""

    query: str
    expected_answer: str
    actual_answer: str
    retrieval_recall: float
    answer_similarity: float
    hallucination_score: float
    confidence: float
    confidence_calibration_error: float
    latency_ms: int
    passed: bool
    category: str = "unknown"
    difficulty: str = "unknown"


@dataclass
class EvaluationSummary:
    """Aggregated summary across all evaluated cases."""

    total_cases: int
    passed_count: int
    failed_count: int
    avg_retrieval_recall: float
    avg_answer_similarity: float
    avg_hallucination_score: float
    avg_confidence_calibration_error: float
    avg_latency_ms: float
    results_by_category: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenisation with basic punctuation stripping."""
    return [
        tok
        for tok in re.sub(r"[^\w\s]", "", text.lower()).split()
        if tok
    ]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on common terminators."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_retrieval_recall(
    retrieved_ids: List[str],
    expected_ids: List[str],
) -> float:
    """Fraction of expected chunk IDs that appear in the retrieved set."""
    if not expected_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for eid in expected_ids if eid in retrieved_set)
    return hits / len(expected_ids)


def compute_answer_similarity(actual: str, expected: str) -> float:
    """Token-overlap ratio (Jaccard-like) between actual and expected answers."""
    actual_tokens = set(_tokenize(actual))
    expected_tokens = set(_tokenize(expected))
    if not actual_tokens and not expected_tokens:
        return 1.0
    if not actual_tokens or not expected_tokens:
        return 0.0
    intersection = actual_tokens & expected_tokens
    union = actual_tokens | expected_tokens
    return len(intersection) / len(union)


def compute_hallucination_score(
    answer: str,
    context_chunks: List[str],
) -> float:
    """Fraction of answer sentences not lexically supported by any context chunk.

    A sentence is considered *supported* when at least half of its
    non-trivial tokens appear in the concatenated context.

    Returns 0.0 when *context_chunks* is empty because hallucination
    cannot be measured without reference context.
    """
    if not context_chunks:
        return 0.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    context_tokens = set(_tokenize(" ".join(context_chunks)))
    unsupported = 0

    for sentence in sentences:
        sent_tokens = _tokenize(sentence)
        if not sent_tokens:
            continue
        overlap = sum(1 for t in sent_tokens if t in context_tokens)
        if overlap < len(sent_tokens) / 2:
            unsupported += 1

    return unsupported / len(sentences)


def compute_confidence_calibration(
    confidence: float,
    answer_similarity: float,
) -> float:
    """Absolute difference between stated confidence and actual similarity."""
    return abs(confidence - answer_similarity)


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------

_SIMILARITY_PASS_THRESHOLD = 0.3
_HALLUCINATION_FAIL_THRESHOLD = 0.5


def evaluate_single(
    case: EvaluationCase,
    actual_answer: str,
    actual_chunks: List[str],
    confidence: float,
    latency_ms: int,
) -> EvaluationResult:
    """Evaluate a single test case and return an *EvaluationResult*."""
    try:
        retrieval_recall = compute_retrieval_recall(
            [c if isinstance(c, str) else str(c) for c in actual_chunks],
            case.expected_chunk_ids or [],
        )
        answer_sim = compute_answer_similarity(actual_answer, case.expected_answer)
        hallucination = compute_hallucination_score(actual_answer, actual_chunks)
        calibration = compute_confidence_calibration(confidence, answer_sim)

        passed = (
            answer_sim >= _SIMILARITY_PASS_THRESHOLD
            and hallucination < _HALLUCINATION_FAIL_THRESHOLD
        )

        result = EvaluationResult(
            query=case.query,
            expected_answer=case.expected_answer,
            actual_answer=actual_answer,
            retrieval_recall=round(retrieval_recall, 4),
            answer_similarity=round(answer_sim, 4),
            hallucination_score=round(hallucination, 4),
            confidence=round(confidence, 4),
            confidence_calibration_error=round(calibration, 4),
            latency_ms=latency_ms,
            passed=passed,
            category=case.category,
            difficulty=case.difficulty,
        )

        log_info(
            f"Evaluated [{case.category}/{case.difficulty}] "
            f"sim={result.answer_similarity} hall={result.hallucination_score} "
            f"passed={passed}"
        )
        return result

    except Exception as exc:
        log_error(f"Evaluation failed for query '{case.query}': {exc}")
        return EvaluationResult(
            query=case.query,
            expected_answer=case.expected_answer,
            actual_answer=actual_answer,
            retrieval_recall=0.0,
            answer_similarity=0.0,
            hallucination_score=1.0,
            confidence=confidence,
            confidence_calibration_error=1.0,
            latency_ms=latency_ms,
            passed=False,
        )


def summarize_results(results: List[EvaluationResult]) -> EvaluationSummary:
    """Aggregate a list of *EvaluationResult* objects into an *EvaluationSummary*."""
    if not results:
        log_warning("summarize_results called with an empty results list")
        return EvaluationSummary(
            total_cases=0,
            passed_count=0,
            failed_count=0,
            avg_retrieval_recall=0.0,
            avg_answer_similarity=0.0,
            avg_hallucination_score=0.0,
            avg_confidence_calibration_error=0.0,
            avg_latency_ms=0.0,
            results_by_category={},
        )

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    def _avg(attr: str) -> float:
        return round(sum(getattr(r, attr) for r in results) / total, 4)

    # Per-category breakdown
    by_category: dict = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    category_summary: dict = {}
    for cat, cat_results in by_category.items():
        n = len(cat_results)
        category_summary[cat] = {
            "count": n,
            "passed": sum(1 for r in cat_results if r.passed),
            "avg_similarity": round(
                sum(r.answer_similarity for r in cat_results) / n, 4
            ),
        }

    summary = EvaluationSummary(
        total_cases=total,
        passed_count=passed,
        failed_count=total - passed,
        avg_retrieval_recall=_avg("retrieval_recall"),
        avg_answer_similarity=_avg("answer_similarity"),
        avg_hallucination_score=_avg("hallucination_score"),
        avg_confidence_calibration_error=_avg("confidence_calibration_error"),
        avg_latency_ms=round(sum(r.latency_ms for r in results) / total, 2),
        results_by_category=category_summary,
    )

    log_info(
        f"Evaluation summary: {passed}/{total} passed, "
        f"avg_sim={summary.avg_answer_similarity}, "
        f"avg_hall={summary.avg_hallucination_score}"
    )
    return summary


# ---------------------------------------------------------------------------
# Sample dataset
# ---------------------------------------------------------------------------

def create_sample_dataset() -> List[EvaluationCase]:
    """Return a diverse set of evaluation cases spanning categories and difficulties."""
    return [
        # --- Factual / Easy ---
        EvaluationCase(
            query="What is photosynthesis?",
            expected_answer=(
                "Photosynthesis is the process by which green plants convert "
                "sunlight into chemical energy, producing glucose and oxygen "
                "from carbon dioxide and water."
            ),
            category="factual",
            difficulty="easy",
        ),
        EvaluationCase(
            query="What is the boiling point of water at sea level?",
            expected_answer="The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit.",
            category="factual",
            difficulty="easy",
        ),
        EvaluationCase(
            query="Who wrote the play Romeo and Juliet?",
            expected_answer="William Shakespeare wrote Romeo and Juliet.",
            category="factual",
            difficulty="easy",
        ),
        # --- Factual / Medium ---
        EvaluationCase(
            query="Explain how mitochondria produce ATP.",
            expected_answer=(
                "Mitochondria produce ATP through oxidative phosphorylation. "
                "The electron transport chain creates a proton gradient across "
                "the inner mitochondrial membrane, and ATP synthase uses that "
                "gradient to synthesise ATP from ADP and inorganic phosphate."
            ),
            expected_chunk_ids=["bio_mito_01", "bio_mito_02"],
            category="factual",
            difficulty="medium",
        ),
        EvaluationCase(
            query="What causes tides on Earth?",
            expected_answer=(
                "Tides are primarily caused by the gravitational pull of the "
                "Moon and, to a lesser extent, the Sun on Earth's oceans."
            ),
            category="factual",
            difficulty="medium",
        ),
        # --- Conceptual / Easy ---
        EvaluationCase(
            query="Why is the sky blue?",
            expected_answer=(
                "The sky appears blue because shorter blue wavelengths of "
                "sunlight are scattered more than other colours by the gases "
                "and particles in Earth's atmosphere, a phenomenon known as "
                "Rayleigh scattering."
            ),
            category="conceptual",
            difficulty="easy",
        ),
        # --- Conceptual / Medium ---
        EvaluationCase(
            query="How does natural selection drive evolution?",
            expected_answer=(
                "Natural selection drives evolution by favouring individuals "
                "with traits that improve survival and reproduction. Over "
                "generations, beneficial traits become more common in the "
                "population while harmful traits decline."
            ),
            expected_chunk_ids=["evo_ns_01"],
            category="conceptual",
            difficulty="medium",
        ),
        EvaluationCase(
            query="What is the difference between TCP and UDP?",
            expected_answer=(
                "TCP is a connection-oriented protocol that guarantees reliable, "
                "ordered delivery of data. UDP is connectionless and does not "
                "guarantee delivery, making it faster but less reliable."
            ),
            category="conceptual",
            difficulty="medium",
        ),
        # --- Conceptual / Hard ---
        EvaluationCase(
            query="Explain the CAP theorem in distributed systems.",
            expected_answer=(
                "The CAP theorem states that a distributed system can "
                "simultaneously provide at most two of the following three "
                "guarantees: Consistency, Availability, and Partition "
                "tolerance. In the presence of a network partition, the system "
                "must choose between consistency and availability."
            ),
            expected_chunk_ids=["cs_cap_01", "cs_cap_02"],
            category="conceptual",
            difficulty="hard",
        ),
        EvaluationCase(
            query="How does backpropagation work in neural networks?",
            expected_answer=(
                "Backpropagation computes the gradient of the loss function "
                "with respect to each weight by applying the chain rule "
                "layer by layer from the output back to the input. These "
                "gradients are then used by an optimiser to update the weights."
            ),
            category="conceptual",
            difficulty="hard",
        ),
        # --- Edge Case ---
        EvaluationCase(
            query="",
            expected_answer="I'm sorry, I didn't receive a question. Could you please try again?",
            category="edge_case",
            difficulty="easy",
        ),
        EvaluationCase(
            query="asdfghjkl random gibberish xyz",
            expected_answer=(
                "I could not understand your question. Please rephrase and try again."
            ),
            category="edge_case",
            difficulty="medium",
        ),
        EvaluationCase(
            query="Tell me everything about everything.",
            expected_answer=(
                "Your question is too broad. Could you narrow it down to a "
                "specific topic so I can provide a useful answer?"
            ),
            category="edge_case",
            difficulty="medium",
        ),
        EvaluationCase(
            query="What is the meaning of life, the universe, and everything?",
            expected_answer=(
                "According to Douglas Adams' The Hitchhiker's Guide to the "
                "Galaxy, the answer is 42. Philosophically, the meaning of "
                "life is a deeply personal question with many perspectives."
            ),
            category="edge_case",
            difficulty="hard",
        ),
    ]
