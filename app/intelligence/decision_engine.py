# app/intelligence/decision_engine.py
"""
Central Decision Engine.
The brain of the pipeline controller - decides which model to use,
whether reasoning is required, which retrieval strategy to apply,
which response mode to use, and whether to trigger tools.
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
    """

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

        Args:
            query: The user's query text
            user_id: User identifier
            query_type: FACTUAL, CONCEPTUAL, COMPARATIVE, PROCEDURAL
            intent: Query intent from classifier
            knowledge_level: beginner, intermediate, advanced
            preferences: User preferences dict
            budget_remaining_pct: Remaining daily budget percentage
        """
        decision = PipelineDecision()
        preferences = preferences or {}
        query_lower = query.lower().strip()

        # ── 1. Model Selection ──
        decision.model_name, decision.max_output_tokens = self._select_model(
            query_type, query_lower, budget_remaining_pct, preferences
        )

        # ── 2. Reasoning Decision ──
        decision.needs_reasoning, decision.reasoning_type = self._decide_reasoning(
            query_lower, query_type
        )

        # ── 3. Retrieval Strategy ──
        decision.retrieval_strategy, decision.top_k_boost, decision.enable_gap_fill = (
            self._decide_retrieval(query_type, query_lower, knowledge_level)
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
    ) -> tuple:
        """Select model based on query complexity and budget."""
        explicit_model = preferences.get("model_name")
        if explicit_model:
            return explicit_model, 600

        # Budget-aware selection
        if budget_remaining_pct < 10:
            return "llama-3.1-8b-instant", 300

        # Complexity-based
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

    def _decide_reasoning(self, query_lower: str, query_type: str) -> tuple:
        """Decide if multi-step reasoning is needed."""
        comparison_keywords = [
            "compare", "contrast", "difference between", "vs", "versus",
            "similarities", "pros and cons",
        ]
        multi_step_signals = [
            query_lower.count("?") > 1,
            " and " in query_lower and any(kw in query_lower for kw in ["how", "what", "why", "explain"]),
            query_type == "COMPARATIVE",
        ]

        if any(kw in query_lower for kw in comparison_keywords):
            return True, "comparison"
        if sum(multi_step_signals) >= 2:
            return True, "multi_step"
        return False, "single"

    def _decide_retrieval(
        self, query_type: str, query_lower: str, knowledge_level: str
    ) -> tuple:
        """Decide retrieval strategy."""
        # Advanced users get deeper retrieval
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
