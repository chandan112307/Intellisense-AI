# app/intelligence/decision_engine.py
"""
Central Decision Engine.
The brain of the pipeline controller - decides which model to use,
whether reasoning is required, which retrieval strategy to apply,
which response mode to use, and whether to trigger tools.

Now adaptive: consults UnifiedLearningSystem to learn from past decisions
and adjust behavior dynamically based on historical outcomes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.logging import log_info


@dataclass
class PipelineDecision:
    """Output of the decision engine - guides the entire pipeline."""
    # Model selection
    model_name: str = "llama-3.1-8b-instant"
    max_output_tokens: int = 400

    # Reasoning
    needs_reasoning: bool = False
    reasoning_type: str = "single"  # single, multi_step, comparison

    # Retrieval strategy
    retrieval_strategy: str = "standard"  # standard, expanded, deep
    top_k_boost: int = 0  # Additional top_k to add
    enable_gap_fill: bool = True

    # Response strategy
    response_mode: str = "teaching"  # teaching, exam, hint, summary, detailed
    response_tone: str = "encouraging"
    include_examples: bool = True
    include_references: bool = True
    max_response_tokens: int = 600

    # Tool usage
    tool_needed: Optional[str] = None  # calculator, code, planner, None

    # Adaptive parameters (from user profile)
    complexity: str = "moderate"  # simple, moderate, detailed
    add_explanations: bool = False

    # Tracing
    decision_reasons: List[str] = field(default_factory=list)


class DecisionEngine:
    """
    Central decision-maker that consults user profile, query analysis,
    cost constraints, and system state to make pipeline decisions.

    Adaptive: uses UnifiedLearningSystem to learn which decisions lead
    to better outcomes and adjusts model/strategy selection over time.
    """

    def __init__(self):
        self._learning = None  # Lazy-loaded to avoid circular imports

    def _get_learning(self):
        """Lazy-load the learning system to avoid circular imports."""
        if self._learning is None:
            try:
                from app.intelligence.unified_learning import get_unified_learning
                self._learning = get_unified_learning()
            except Exception:
                self._learning = None
        return self._learning

    def decide(
        self,
        query: str,
        user_id: str,
        query_type: str = "",
        intent: str = "",
        knowledge_level: str = "intermediate",
        preferences: Optional[Dict[str, Any]] = None,
        budget_remaining_pct: float = 100.0,
    ) -> PipelineDecision:
        """
        Make all pipeline decisions based on available signals.

        Consults historical learning data to adapt decisions dynamically.
        """
        decision = PipelineDecision()
        preferences = preferences or {}
        query_lower = query.lower().strip()
        learning = self._get_learning()

        # ── 1. Model Selection (adaptive) ──
        decision.model_name, decision.max_output_tokens = self._select_model(
            query_type, query_lower, budget_remaining_pct, preferences, learning
        )

        # ── 2. Reasoning Decision (adaptive) ──
        decision.needs_reasoning, decision.reasoning_type = self._decide_reasoning(
            query_lower, query_type, learning
        )

        # ── 3. Retrieval Strategy (adaptive) ──
        decision.retrieval_strategy, decision.top_k_boost, decision.enable_gap_fill = (
            self._decide_retrieval(query_type, query_lower, knowledge_level, learning)
        )

        # ── 4. Response Strategy ──
        (
            decision.response_mode,
            decision.response_tone,
            decision.include_examples,
            decision.include_references,
            decision.max_response_tokens,
        ) = self._decide_response(query_lower, knowledge_level, preferences)

        # ── 5. Tool Detection ──
        decision.tool_needed = self._detect_tool_need(query_lower)

        # ── 6. User-Adaptive Parameters ──
        decision.complexity, decision.add_explanations = self._adapt_to_user(
            knowledge_level
        )

        log_info(
            f"Decision: model={decision.model_name}, reasoning={decision.needs_reasoning}, "
            f"retrieval={decision.retrieval_strategy}, response={decision.response_mode}, "
            f"tool={decision.tool_needed}, complexity={decision.complexity}"
        )
        return decision

    def _select_model(
        self,
        query_type: str,
        query_lower: str,
        budget_remaining_pct: float,
        preferences: Dict[str, Any],
        learning,
    ) -> tuple:
        """Select model based on query complexity, budget, and learned performance."""
        explicit_model = preferences.get("model_name")
        if explicit_model:
            return explicit_model, 600

        # Budget-aware selection
        if budget_remaining_pct < 10:
            return "llama-3.1-8b-instant", 300

        # ── Adaptive: check if learning system recommends a model ──
        if learning and query_type:
            learned_model = learning.get_best_model_for_query_type(query_type)
            if learned_model:
                log_info(f"Adaptive model selection: learned best={learned_model} for {query_type}")
                return learned_model, 600

            # Check if failures suggest upgrading
            if learning.should_upgrade_model(query_type):
                log_info(f"Adaptive: failure rate high for {query_type}, upgrading model")
                return "llama-3.1-70b-versatile", 600

        # ── Static fallback ──
        is_complex = query_type in ("COMPARATIVE", "PROCEDURAL") or any(
            kw in query_lower
            for kw in ["compare", "explain in detail", "analyze", "evaluate", "comprehensive"]
        )
        is_verification = any(
            kw in query_lower for kw in ["verify", "fact check", "is it true"]
        )

        if is_verification:
            return "llama-3.1-70b-versatile", 600
        if is_complex:
            return "llama-3.1-70b-versatile", 600

        return "llama-3.1-8b-instant", 400

    def _decide_reasoning(self, query_lower: str, query_type: str, learning) -> tuple:
        """Decide if multi-step reasoning is needed, using historical patterns."""
        comparison_keywords = [
            "compare", "contrast", "difference between", "vs", "versus",
            "similarities", "pros and cons",
        ]
        multi_step_signals = [
            query_lower.count("?") > 1,
            " and " in query_lower and any(kw in query_lower for kw in ["how", "what", "why", "explain"]),
            query_type == "COMPARATIVE",
        ]

        # ── Adaptive: check learned best reasoning type ──
        if learning and query_type:
            learned_reasoning = learning.get_best_reasoning_type(query_type)
            if learned_reasoning and learned_reasoning != "single":
                log_info(f"Adaptive reasoning: learned best={learned_reasoning} for {query_type}")
                return True, learned_reasoning

        if any(kw in query_lower for kw in comparison_keywords):
            return True, "comparison"
        if sum(multi_step_signals) >= 2:
            return True, "multi_step"
        return False, "single"

    def _decide_retrieval(
        self, query_type: str, query_lower: str, knowledge_level: str, learning
    ) -> tuple:
        """Decide retrieval strategy, adaptively adjusting from learned data."""
        # ── Adaptive: check learned best strategy ──
        if learning and query_type:
            learned_strategy = learning.get_best_strategy_for_query_type(query_type)
            if learned_strategy:
                log_info(f"Adaptive retrieval: learned best={learned_strategy} for {query_type}")
                boost = 3 if learned_strategy == "expanded" else (8 if learned_strategy == "deep" else 0)
                return learned_strategy, boost, True

            # Adaptive top_k adjustment
            adaptive_boost = learning.get_adaptive_top_k(query_type, base_top_k=0)
            if adaptive_boost > 0:
                log_info(f"Adaptive top_k boost: +{adaptive_boost} for {query_type}")

        # ── Static fallback ──
        if knowledge_level == "advanced":
            return "expanded", 3, True
        if query_type in ("COMPARATIVE", "PROCEDURAL"):
            return "expanded", 5, True
        if any(kw in query_lower for kw in ["detail", "comprehensive", "everything about"]):
            return "deep", 8, True
        return "standard", 0, True

    def _decide_response(
        self, query_lower: str, knowledge_level: str, preferences: Dict[str, Any]
    ) -> tuple:
        """Decide response mode and parameters."""
        explicit_mode = preferences.get("response_mode")
        if explicit_mode:
            configs = {
                "teaching": ("teaching", "encouraging", True, True, 600),
                "exam": ("exam", "neutral", False, False, 200),
                "hint": ("hint", "encouraging", False, False, 100),
                "summary": ("summary", "neutral", False, True, 150),
                "detailed": ("detailed", "formal", True, True, 1000),
            }
            if explicit_mode in configs:
                return configs[explicit_mode]

        if any(kw in query_lower for kw in ["quiz me", "test me", "examine"]):
            return "exam", "neutral", False, False, 200
        if any(kw in query_lower for kw in ["hint", "clue", "nudge"]):
            return "hint", "encouraging", False, False, 100
        if any(kw in query_lower for kw in ["summarize", "summary", "brief", "tldr"]):
            return "summary", "neutral", False, True, 150
        if any(kw in query_lower for kw in ["explain in detail", "elaborate", "deep dive"]):
            return "detailed", "formal", True, True, 1000

        if knowledge_level == "beginner":
            return "teaching", "encouraging", True, True, 600
        return "teaching", "encouraging", True, True, 600

    def _detect_tool_need(self, query_lower: str) -> Optional[str]:
        """Detect if query needs a tool."""
        import re
        math_pattern = r'[\d]+\s*[\+\-\*/\^%]\s*[\d]+'
        if re.search(math_pattern, query_lower):
            return "calculator"
        if any(kw in query_lower for kw in ["calculate", "compute", "solve", "equation"]):
            return "calculator"
        if any(kw in query_lower for kw in ["run code", "execute", "write a program"]):
            return "code"
        if any(kw in query_lower for kw in ["plan", "steps to", "how to build", "roadmap"]):
            return "planner"
        return None

    def _adapt_to_user(self, knowledge_level: str) -> tuple:
        """Adapt parameters based on user knowledge level."""
        if knowledge_level == "beginner":
            return "simple", True
        if knowledge_level == "advanced":
            return "detailed", False
        return "moderate", False
