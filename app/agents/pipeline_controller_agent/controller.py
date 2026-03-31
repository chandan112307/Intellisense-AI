# app/agents/pipeline_controller_agent/controller.py
from app.agents.query_understanding_agent.agent import QueryUnderstandingAgent
from app.agents.retrieval_agent.orchestrator import RetrievalOrchestratorAgent
from app.agents.response_synthesizer_agent.synthesizer import ResponseSynthesizer

from app.agents.query_understanding_agent.schema import (
    QueryUnderstandingInput,
    QueryUnderstandingOutput,
    Preferences
)

# IMPORTANT FIX: Import RetrievalParams from retrieval agent schema
from app.agents.retrieval_agent.schema import (
    RetrievalInput,
    RetrievalOutput,
    RetrievalParams as RetrievalParams_RetrievalAgent,
)

from app.agents.response_synthesizer_agent.schema import SynthesisInput, SynthesisOutput

# EviLearn Imports
from app.agents.claim_extraction_agent.agent import ClaimExtractionAgent
from app.agents.claim_extraction_agent.schema import ClaimExtractionInput
from app.agents.verification_agent.agent import VerificationAgent
from app.agents.verification_agent.schema import VerificationInput
from app.agents.explanation_agent.agent import ExplanationAgent
from app.agents.explanation_agent.schema import ExplanationInput


from .controller_config import (
    DEFAULT_RETRIEVERS,
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_OUTPUT_TOKENS,
    FALLBACK_INSUFFICIENT_CONTEXT,
    FALLBACK_QUERY_UNDERSTANDING_FAILURE,
    FALLBACK_SYNTHESIZER_FAILURE,
    FALLBACK_NO_KNOWLEDGE,
    NO_KNOWLEDGE_SCORE_THRESHOLD,
)

import time
import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from app.core.logging import log_info, log_error
from app.rag.intent_classifier import classify_intent, IntentResult, QueryIntent
from app.rag.query_rewriter import rewrite_query
from app.rag.retrieval_validator import validate_retrieval
from app.rag.subject_detector import detect_subject
from app.rag.context_verifier import verify_context
from app.rag.query_expander import rewrite_for_retry
from app.core.config import CONTEXT_VERIFICATION_ENABLED, GROUNDED_MODE_THRESHOLD, FAILURE_PREDICTION_ENABLED
from app.rag.failure_predictor import predict_failure

# ── Integrated Intelligence Systems ──
from app.intelligence.decision_engine import DecisionEngine, PipelineDecision
from app.intelligence.unified_learning import get_unified_learning
from app.intelligence.user_profile import get_user_profile, get_adaptive_params, record_interaction
from app.observability.metrics import record_query as record_metrics
from app.cost.tracker import CostTracker, record_usage, select_model as cost_select_model
from app.reasoning.engine import needs_multi_step, reason as run_reasoning
from app.strategy.response_strategy import build_strategy_prompt_prefix, ResponseStrategy


class PipelineControllerAgent:
    def __init__(
        self,
        query_understander: QueryUnderstandingAgent,
        retriever_orchestrator: RetrievalOrchestratorAgent,
        response_synthesizer: ResponseSynthesizer,
        claim_extractor: ClaimExtractionAgent = None,
        verifier: VerificationAgent = None,
        explainer: ExplanationAgent = None
    ):
        # Load .env file with encoding fallback
        try:
            load_dotenv()
        except UnicodeDecodeError:
            # Try UTF-16 encoding (common on Windows)
            try:
                load_dotenv(encoding='utf-16')
            except Exception:
                # Try UTF-16 with BOM
                try:
                    load_dotenv(encoding='utf-16-le')
                except Exception:
                    # If all encodings fail, continue without .env file
                    pass
        self.query_understander = query_understander
        self.retriever_orchestrator = retriever_orchestrator
        self.response_synthesizer = response_synthesizer
        self.claim_extractor = claim_extractor
        self.verifier = verifier
        self.explainer = explainer

        self.default_retrievers = DEFAULT_RETRIEVERS
        self.default_model = DEFAULT_MODEL_NAME
        self.default_max_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        self.decision_engine = DecisionEngine()
        self.learning_system = get_unified_learning()

    async def run(
        self,
        query: str,
        user_id: str,
        session_id: str,
        preferences: Preferences,
        conversation_history: List,
        allow_agentic: bool = False,
        model_name: str = "llama-3.1-8b-instant"
    ) -> Dict[str, Any]:

        start_time = time.time()
        warnings: List[str] = []
        trace: Dict[str, Any] = {}

        # ── DECISION ENGINE: Central intelligence decisions ──
        try:
            user_profile = get_user_profile(user_id)
            knowledge_level = user_profile.get("knowledge_level", "intermediate")
            adaptive_params = get_adaptive_params(user_id)
        except Exception:
            knowledge_level = "intermediate"
            adaptive_params = {"top_k": 5, "complexity": "moderate", "add_explanations": False}

        try:
            cost_tracker = CostTracker()
            budget_status = cost_tracker.get_budget_status()
            budget_pct = 100.0 - budget_status.get("utilization_pct", 0.0)
        except Exception:
            budget_pct = 100.0

        decision = self.decision_engine.decide(
            query=query,
            user_id=user_id,
            knowledge_level=knowledge_level,
            preferences=preferences if isinstance(preferences, dict) else {},
            budget_remaining_pct=budget_pct,
        )

        # Apply decision engine outputs
        model_name = model_name or decision.model_name

        trace["decision_engine"] = {
            "model": decision.model_name,
            "needs_reasoning": decision.needs_reasoning,
            "reasoning_type": decision.reasoning_type,
            "retrieval_strategy": decision.retrieval_strategy,
            "response_mode": decision.response_mode,
            "tool_needed": decision.tool_needed,
            "complexity": decision.complexity,
            "knowledge_level": knowledge_level,
        }

        prefs = Preferences(
            response_style=preferences.get("response_style", "concise"),
            max_length=preferences.get("max_length", 300),
            domain=preferences.get("domain", "general")
        )

        # ======================
        # 1. QUERY UNDERSTANDING
        # ======================
        try:
            self.query_understanding_input = QueryUnderstandingInput(
                query_text=query,
                user_id=user_id,
                session_id=session_id,
                preferences=prefs,
                conversation_history=conversation_history,
                timestamp=datetime.datetime.now()
            )
            self.query_understanding_output: QueryUnderstandingOutput = \
                await self.query_understander.run(self.query_understanding_input)

            trace["query_understanding"] = {
                "rewritten_query": self.query_understanding_output.rewritten_query,
                "intent": self.query_understanding_output.intent,
                "retrievers_to_use": self.query_understanding_output.retrievers_to_use,
                "retrieval_params": self.query_understanding_output.retrieval_params.model_dump(),
                "style_preferences": self.query_understanding_output.style_preferences.model_dump(),
            }

        except Exception as e:
            from app.agents.query_understanding_agent.schema import RetrievalParams, StylePreferences

            warnings.append(f"QueryUnderstanding failed: {e}")

            default_retrieval_params = RetrievalParams(
                top_k_vector=6,
                top_k_keyword=4,
                top_k_web=3,
                top_k_youtube=1
            )
            default_style = StylePreferences(
                type="concise",
                tone="neutral"
            )

            self.query_understanding_output = QueryUnderstandingOutput(
                intent="none",
                rewritten_query=query,
                retrievers_to_use=self.default_retrievers,
                retrieval_params=default_retrieval_params,
                style_preferences=default_style
            )

            trace["query_understanding"] = {
                "rewritten_query": query,
                "intent": "none",
                "retrievers_to_use": self.default_retrievers,
                "retrieval_params": default_retrieval_params.model_dump(),
                "style_preferences": default_style.model_dump(),
            }

        # ======================
        # 1.5 INTENT CLASSIFICATION (rule-based, no LLM)
        # ======================
        intent_result = classify_intent(query)
        log_info(
            f"Intent classified: {intent_result.intent.value}, "
            f"section={intent_result.target_section}, "
            f"confidence={intent_result.confidence}"
        )

        trace["intent_classification"] = {
            "intent": intent_result.intent.value,
            "target_section": intent_result.target_section,
            "has_document_reference": intent_result.has_document_reference,
            "confidence": intent_result.confidence,
            "explanation": intent_result.explanation,
        }

        # ======================
        # 1.6 QUERY REWRITING (deterministic, no LLM)
        # ======================
        rewritten_for_retrieval = rewrite_query(
            raw_query=query,
            intent_result=intent_result,
        )
        # Use intent-aware rewrite if the LLM rewrite is too short or generic
        llm_rewrite = self.query_understanding_output.rewritten_query
        if len(rewritten_for_retrieval.split()) > len(llm_rewrite.split()):
            effective_query = rewritten_for_retrieval
        else:
            effective_query = llm_rewrite

        log_info(f"Effective retrieval query: '{effective_query}'")
        trace["query_rewriting"] = {
            "llm_rewrite": llm_rewrite,
            "intent_rewrite": rewritten_for_retrieval,
            "effective_query": effective_query,
        }

        # ======================
        # 1.7 SUBJECT SCOPE DETECTION (rule-based, no LLM)
        # ======================
        # Import SubjectScope for fallback creation
        from app.rag.subject_detector import detect_subject, SubjectScope
        subject_scope = detect_subject(query)
        
        # FIX: Confidence-Aware Subject Filtering
        # If subject confidence is low, we disable strict filtering to avoid missing relevant docs.
        SUBJECT_CONFIDENCE_THRESHOLD = 0.45
        search_scope = subject_scope
        
        if subject_scope.subject and subject_scope.confidence < SUBJECT_CONFIDENCE_THRESHOLD:
            log_info(f"Subject '{subject_scope.subject}' low confidence ({subject_scope.confidence:.2f} < {SUBJECT_CONFIDENCE_THRESHOLD}). Disabling strict filtering.")
            search_scope = SubjectScope(
                subject="", # Disable filtering
                confidence=subject_scope.confidence,
                is_ambiguous=True,
                content_type=subject_scope.content_type,
                secondary_subject=subject_scope.secondary_subject
            )
            warnings.append("low_conf_subject_filter_disabled")

        log_info(
            f"Subject scope: detected='{subject_scope.subject}', "
            f"used='{search_scope.subject}', "
            f"confidence={subject_scope.confidence:.2f}"
        )
        
        trace["subject_scope"] = subject_scope.model_dump()
        if subject_scope.is_ambiguous:
            warnings.append("ambiguous_subject_scope")

        # ======================
        # 1.8 QUERY STRUCTURAL CLASSIFICATION (Advanced)
        # ======================
        # ======================
        # 1.8 QUERY STRUCTURAL CLASSIFICATION (Advanced)
        # ======================
        from app.rag.query_classifier import query_classifier, QueryType
        query_type_result = query_classifier.classify(query)
        log_info(f"Query Structure: {query_type_result.query_type.value} (conf={query_type_result.confidence})")
        trace["query_structure"] = query_type_result.model_dump()
        
        # FIX: Conceptual Query Handling
        # If query is conceptual, we expand it with definition keywords to boost semantic matching of explanation paragraphs.
        is_conceptual = False
        if query_type_result.query_type == QueryType.CONCEPTUAL:
            is_conceptual = True
            # Appending definition anchors directly to the retrieval query
            # This helps vector search find "definition of X" or "concept of X"
            if "definition" not in effective_query.lower() and "concept" not in effective_query.lower():
                effective_query += " definition concept principle architecture"
                log_info(f"Conceptual query detected. Expanded query: '{effective_query}'")
        
        # Future: Use query_type to adjust retrieval strategy
        # ── Refine decision with query_type ──
        decision = self.decision_engine.decide(
            query=query,
            user_id=user_id,
            query_type=query_type_result.query_type.value,
            intent=intent_result.intent.value,
            knowledge_level=knowledge_level,
            preferences=preferences if isinstance(preferences, dict) else {},
            budget_remaining_pct=budget_pct,
        )
        trace["decision_engine"]["query_type"] = query_type_result.query_type.value
        trace["decision_engine"]["intent"] = intent_result.intent.value
        
        # ======================
        # 2. RETRIEVAL (section-aware + subject-scoped)
        # ======================
        try:
            # 🔥 CRITICAL FIX: convert QueryUnderstanding RetrievalParams → RetrievalAgent RetrievalParams
            converted_params = RetrievalParams_RetrievalAgent(
                **self.query_understanding_output.retrieval_params.model_dump()
            )

            self.retrieval_agent_input = RetrievalInput(
                user_id=user_id,
                session_id=session_id,
                rewritten_query=effective_query,  # Use intent-aware query (potentially expanded)
                retrievers_to_use=self.query_understanding_output.retrievers_to_use,
                retrieval_params=converted_params,
                conversation_history=conversation_history,
                preferences=preferences,
                is_conceptual=is_conceptual # Pass flag to orchestrator
            )

            self.retrieval_agent_output: RetrievalOutput = \
                await self.retriever_orchestrator.run(
                    self.retrieval_agent_input,
                    intent_result=intent_result,  # Pass intent for section-aware retrieval
                    subject_scope=search_scope,  # Pass potentially broadened scope
                    query_type=query_type_result.query_type.value,  # Pass for dynamic weighting
                    student_id=user_id,  # Enable student knowledge retrieval
                )
            
            # FIX: Automatic Fallback for Subject Filtering
            # If we applied a subject filter and got ZERO results, strictly fallback to global search.
            if search_scope.subject and len(self.retrieval_agent_output.chunks) == 0:
                log_info("Subject-scoped retrieval returned 0 chunks. TRIGGERING FALLBACK: Global Search.")
                warnings.append("subject_filter_fallback_triggered")
                
                fallback_scope = SubjectScope(
                    subject="", 
                    is_ambiguous=True,
                    content_type=search_scope.content_type
                )
                
                # Retry with global scope
                self.retrieval_agent_output = await self.retriever_orchestrator.run(
                    self.retrieval_agent_input,
                    intent_result=intent_result,
                    subject_scope=fallback_scope,
                    query_type=query_type_result.query_type.value,
                    student_id=user_id,
                )
                
                # Update trace to indicate fallback happened
                if self.retrieval_agent_output.retrieval_trace:
                     self.retrieval_agent_output.retrieval_trace["fallback_triggered"] = True

        except Exception as e:
            warnings.append(f"Retrieval failed: {e}")
            self.retrieval_agent_output = RetrievalOutput(
                chunks=[],
                retrieval_trace={},
                trace_id=f"fallback-{int(time.time() * 1000)}",
            )

        # ======================
        # 2.5 POST-RETRIEVAL VALIDATION
        # ======================
        validation = validate_retrieval(
            chunks=self.retrieval_agent_output.chunks,
            intent_result=intent_result,
        )

        # ======================
        # 2.5 DYNAMIC RETRIEVAL LOOP (Agentic Orchestration)
        # ======================
        # We loop to improve retrieval quality if the initial attempt is weak.
        # Strategies:
        # 1. Initial attempt (Strict subject, standard params)
        # 2. Query Relaxation (broader query, lower thresholds)
        # 3. Parameter Expansion (higher top_k)
        
        max_attempts = 3
        attempt = 1
        best_output = self.retrieval_agent_output
        best_validation = validation
        
        while attempt < max_attempts and not best_validation.is_valid and best_validation.should_retry:
            if len(self.retrieval_agent_output.chunks) == 0:
                # If we found NOTHING, a simple retry might not help unless we change the query significantly.
                # For now, we break to avoid wasted cycles, unless we want to try a very broad fallback.
                break

            log_info(f"Retrieval attempt {attempt} failed validation: {best_validation.reason}. Starting attempt {attempt+1}...")
            attempt += 1
            
            # Dynamic Strategy Selection based on failure reason
            # strategy = "relax_query"
            
            try:
                # Use enhanced retry rewrites from query_expander
                retry_rewrites = rewrite_for_retry(effective_query, attempt=attempt)
                retry_query = retry_rewrites[0] if retry_rewrites else rewritten_for_retrieval + " detailed context"
                
                # Relax retrieval parameters (fetch more candidates)
                relaxed_params = converted_params.model_copy()
                relaxed_params.top_k_vector += 5
                relaxed_params.top_k_keyword += 3

                retry_input = RetrievalInput(
                    user_id=user_id,
                    session_id=session_id,
                    rewritten_query=retry_query,
                    retrievers_to_use=self.query_understanding_output.retrievers_to_use,
                    retrieval_params=relaxed_params,
                    conversation_history=conversation_history,
                    preferences=preferences
                )
                
                # We reuse the same intent/subject for now, but in a more advanced version we could relax those too.
                retry_output = await self.retriever_orchestrator.run(
                    retry_input, 
                    intent_result=intent_result,
                    subject_scope=subject_scope,
                    query_type=query_type_result.query_type.value,
                    student_id=user_id,
                )
                
                retry_validation = validate_retrieval(
                    chunks=retry_output.chunks,
                    intent_result=intent_result,
                )
                
                # Greedy selection: if it looks better, take it
                if retry_validation.top_score > best_validation.top_score:
                    log_info(f"Attempt {attempt} improved score: {best_validation.top_score:.3f} -> {retry_validation.top_score:.3f}")
                    self.retrieval_agent_output = retry_output
                    best_validation = retry_validation
                    converted_params = relaxed_params # Update params for trace if needed
                else:
                    log_info(f"Attempt {attempt} did not improve score ({retry_validation.top_score:.3f} vs {best_validation.top_score:.3f}).")
            
            except Exception as e:
                warnings.append(f"Retrieval attempt {attempt} error: {e}")
                log_error(f"Retrieval loop error: {e}")
                break

        validation = best_validation # Ensure final validation is the best one


        trace["retrieval_validation"] = {
            "is_valid": validation.is_valid,
            "reason": validation.reason,
            "top_score": validation.top_score,
            "section_match_count": validation.section_match_count,
            "document_match_count": validation.document_match_count,
        }

        trace["retrieval_trace"] = self.retrieval_agent_output.retrieval_trace

        # ======================
        # 2.6 EARLY NO-KNOWLEDGE EXIT
        # ======================
        # If retrieval found nothing relevant, skip synthesis entirely
        # to avoid slow LLM calls that will just produce an error.
        has_no_chunks = len(self.retrieval_agent_output.chunks) == 0
        has_low_relevance = validation.top_score < NO_KNOWLEDGE_SCORE_THRESHOLD

        if has_no_chunks or has_low_relevance:
            latency_ms = int((time.time() - start_time) * 1000)
            log_info(
                f"No-knowledge fast path: chunks={len(self.retrieval_agent_output.chunks)}, "
                f"top_score={validation.top_score:.3f}, latency={latency_ms}ms"
            )
            return {
                "answer": FALLBACK_NO_KNOWLEDGE,
                "confidence": 0.0,
                "warnings": warnings + ["no_relevant_knowledge"],
                "used_chunk_ids": [],
                "retrieval_trace": trace.get("retrieval_trace", {}),
                "query_understanding": trace.get("query_understanding", {}),
                "trace_id": self.retrieval_agent_output.trace_id,
                "latency_ms": latency_ms,
                "raw_model_output": None,
                "metrics": {},
            }

        # ======================
        # 2.7 CONTEXT VERIFICATION (NEW)
        # ======================
        grounded_only = False
        retrieval_confidence_score = 0.0

        if CONTEXT_VERIFICATION_ENABLED and len(self.retrieval_agent_output.chunks) > 0:
            context_verification = verify_context(
                query=effective_query,
                chunks=self.retrieval_agent_output.chunks,
            )
            trace["context_verification"] = {
                "is_sufficient": context_verification.is_sufficient,
                "coverage_score": context_verification.coverage_score,
                "answer_signal": context_verification.answer_signal_score,
                "has_contradictions": context_verification.has_contradictions,
                "evidence_strength": context_verification.evidence_strength,
                "recommendation": context_verification.recommendation,
                "uncovered_concepts": context_verification.uncovered_concepts[:5],
            }

            # Extract retrieval confidence from orchestrator trace
            ret_trace = self.retrieval_agent_output.retrieval_trace or {}
            ret_confidence = ret_trace.get("retrieval_confidence", {})
            retrieval_confidence_score = ret_confidence.get("score", 0.0)

            # Determine grounded mode
            if context_verification.recommendation == "grounded_only":
                grounded_only = True
                warnings.append("grounded_mode_context_verification")
            elif retrieval_confidence_score < GROUNDED_MODE_THRESHOLD:
                grounded_only = True
                warnings.append("grounded_mode_low_confidence")

            if grounded_only:
                log_info(f"GROUNDED MODE activated: context_rec={context_verification.recommendation}, ret_confidence={retrieval_confidence_score:.3f}")

        # ======================
        # 2.8 FAILURE PREDICTION (Pre-Synthesis Guard)
        # ======================
        if FAILURE_PREDICTION_ENABLED and len(self.retrieval_agent_output.chunks) > 0:
            try:
                ctx_ver = context_verification if CONTEXT_VERIFICATION_ENABLED and len(self.retrieval_agent_output.chunks) > 0 else None
                failure_prediction = predict_failure(
                    chunks=self.retrieval_agent_output.chunks,
                    query=effective_query,
                    context_verification=ctx_ver,
                )
                trace["failure_prediction"] = {
                    "risk_level": failure_prediction.risk_level,
                    "should_retry": failure_prediction.should_retry,
                    "should_expand": failure_prediction.should_expand,
                    "should_ground": failure_prediction.should_ground,
                    "signals": failure_prediction.signals,
                    "explanation": failure_prediction.explanation,
                }

                # Act on failure prediction
                if failure_prediction.should_ground and not grounded_only:
                    grounded_only = True
                    warnings.append("grounded_mode_failure_prediction")
                    log_info(f"FAILURE PREDICTION: grounded mode activated (risk={failure_prediction.risk_level:.3f})")
                elif failure_prediction.should_retry:
                    log_info(f"FAILURE PREDICTION: retry recommended (risk={failure_prediction.risk_level:.3f}: {failure_prediction.explanation})")
                    warnings.append(f"failure_prediction_retry_recommended")

            except Exception as e:
                log_info(f"Failure prediction skipped: {e}")

        # ======================
        # 2.9 REASONING ENGINE (if decision engine says it's needed)
        # ======================
        reasoning_result = None
        if decision.needs_reasoning and len(self.retrieval_agent_output.chunks) > 0:
            try:
                reasoning_result = run_reasoning(
                    query=effective_query,
                    context_chunks=self.retrieval_agent_output.chunks,
                )
                trace["reasoning"] = {
                    "is_multi_step": reasoning_result.is_multi_step,
                    "steps_count": len(reasoning_result.steps),
                    "confidence": reasoning_result.confidence,
                    "reasoning_chain": reasoning_result.reasoning_chain[:500],
                }
                log_info(f"Reasoning complete: {len(reasoning_result.steps)} steps, confidence={reasoning_result.confidence:.3f}")
            except Exception as e:
                log_info(f"Reasoning engine skipped: {e}")

        # ======================
        # 2.10 RESPONSE STRATEGY (from decision engine)
        # ======================
        strategy_prefix = ""
        try:
            strategy = ResponseStrategy(
                mode=decision.response_mode,
                tone=decision.response_tone,
                include_examples=decision.include_examples,
                include_references=decision.include_references,
                max_length_hint=decision.max_response_tokens,
            )
            strategy_prefix = build_strategy_prompt_prefix(strategy)
            trace["response_strategy"] = {
                "mode": decision.response_mode,
                "tone": decision.response_tone,
                "max_tokens": decision.max_response_tokens,
            }
        except Exception as e:
            log_info(f"Strategy prefix skipped: {e}")

        # ======================
        # 2.11 TOOL EXECUTION (if needed)
        # ======================
        tool_result = None
        if decision.tool_needed:
            try:
                from app.tools.tool_registry import get_default_registry
                registry = get_default_registry()
                tool_result = registry.execute_tool(decision.tool_needed, query)
                if tool_result and tool_result.success:
                    trace["tool_execution"] = {
                        "tool": decision.tool_needed,
                        "output": tool_result.output[:200],
                        "success": True,
                    }
                    log_info(f"Tool executed: {decision.tool_needed} -> success")
            except Exception as e:
                log_info(f"Tool execution skipped: {e}")

        # ======================
        # 3. SYNTHESIS (with integrated intelligence)
        # ======================
        # Build enhanced query with reasoning chain and strategy prefix
        synthesis_query = self.query_understanding_output.rewritten_query
        if reasoning_result and reasoning_result.reasoning_chain:
            synthesis_query = f"{strategy_prefix}\n\nReasoning context:\n{reasoning_result.reasoning_chain}\n\nOriginal query: {synthesis_query}"
        elif strategy_prefix:
            synthesis_query = f"{strategy_prefix}\n\n{synthesis_query}"

        # Add tool output if available
        if tool_result and tool_result.success:
            synthesis_query += f"\n\nTool ({decision.tool_needed}) output: {tool_result.output}"

        effective_max_tokens = decision.max_output_tokens
        if isinstance(preferences, dict):
            effective_max_tokens = preferences.get("max_tokens", decision.max_output_tokens)

        try:
            self.response_synthesizer_agent_input = SynthesisInput(
                trace_id=self.retrieval_agent_output.trace_id,
                user_id=user_id,
                session_id=session_id,
                query=synthesis_query,
                conversation_history=conversation_history,
                preferences=preferences,
                model_name=model_name,
                max_output_tokens=effective_max_tokens,
                retrieved_chunks=self.retrieval_agent_output.chunks,
                grounded_only=grounded_only,
                retrieval_confidence=retrieval_confidence_score,
            )
            self.response_synthesizer_agent_output: SynthesisOutput = \
                await self.response_synthesizer.run(self.response_synthesizer_agent_input)

        except Exception as e:
            warnings.append(f"Synthesizer failed: {e}")
            self.response_synthesizer_agent_output = SynthesisOutput(
                answer=FALLBACK_SYNTHESIZER_FAILURE,
                used_chunk_ids=[],
                trace_id=self.retrieval_agent_output.trace_id,
                confidence=0.0,
                warnings=[str(e)],
                raw_model_output=None,
                reasoning=None,
                metrics={},
            )

        if self.response_synthesizer_agent_output.answer == "INSUFFICIENT_CONTEXT":
            warnings.append("Synthesizer reported INSUFFICIENT_CONTEXT.")
            self.response_synthesizer_agent_output.answer = FALLBACK_INSUFFICIENT_CONTEXT
            self.response_synthesizer_agent_output.confidence = 0.0

        # ======================
        # FINAL OUTPUT + POST-PIPELINE INTELLIGENCE
        # ======================
        latency_ms = int((time.time() - start_time) * 1000)
        confidence = self.response_synthesizer_agent_output.confidence
        success = confidence > 0.0 and "INSUFFICIENT" not in self.response_synthesizer_agent_output.answer

        # ── Record to Unified Learning System ──
        try:
            chunk_types = list(set(
                getattr(c, "section_type", None) or (getattr(c, "metadata", {}) or {}).get("section_type", "body")
                for c in self.retrieval_agent_output.chunks[:10]
            ))
            self.learning_system.record_retrieval_outcome(
                query=effective_query,
                query_type=query_type_result.query_type.value if hasattr(query_type_result, "query_type") else "",
                chunk_types=chunk_types,
                confidence=confidence,
                recommendation="proceed" if success else "retry",
                outcome_quality=confidence,
            )
        except Exception as e:
            log_info(f"Learning system record skipped: {e}")

        # ── Record Observability Metrics ──
        try:
            record_metrics(
                latency_ms=latency_ms,
                success=success,
                grounded_mode=grounded_only,
                query_type=query_type_result.query_type.value if hasattr(query_type_result, "query_type") else "unknown",
                tokens_used=self.response_synthesizer_agent_output.metrics.get("tokens", 0) if self.response_synthesizer_agent_output.metrics else 0,
            )
        except Exception as e:
            log_info(f"Metrics recording skipped: {e}")

        # ── Record Cost Tracking ──
        try:
            token_count = self.response_synthesizer_agent_output.metrics.get("tokens", 0) if self.response_synthesizer_agent_output.metrics else 0
            if token_count > 0:
                record_usage(model_name, token_count // 2, token_count // 2)
        except Exception as e:
            log_info(f"Cost tracking skipped: {e}")

        # ── Record User Interaction for Adaptive Learning ──
        try:
            record_interaction(
                user_id=user_id,
                query=query,
                topic=query_type_result.query_type.value if hasattr(query_type_result, "query_type") else "general",
                was_correct=success,
                difficulty=decision.complexity,
            )
        except Exception as e:
            log_info(f"User interaction record skipped: {e}")

        final_payload = {
            "answer": self.response_synthesizer_agent_output.answer,
            "confidence": self.response_synthesizer_agent_output.confidence,
            "warnings": warnings + self.response_synthesizer_agent_output.warnings,
            "used_chunk_ids": self.response_synthesizer_agent_output.used_chunk_ids,
            "retrieval_trace": trace.get("retrieval_trace", {}),
            "query_understanding": trace.get("query_understanding", {}),
            "trace_id": self.response_synthesizer_agent_output.trace_id,
            "latency_ms": latency_ms,
            "raw_model_output": self.response_synthesizer_agent_output.raw_model_output,
            "metrics": self.response_synthesizer_agent_output.metrics,
        }

        return final_payload

    async def run_verification_flow(
        self,
        text: str,
        user_id: str,
        session_id: str,
        preferences: Preferences = None
    ) -> Dict[str, Any]:
        """
        Orchestrates the Claim Verification Flow (EviLearn).
        """
        log_info(f"Starting Verification Flow for user={user_id}")
        start_time = time.time()
        
        # 1. Claim Extraction
        log_info("Step 1: Extracting Claims...")
        extraction_output = await self.claim_extractor.run(ClaimExtractionInput(text=text))
        claims = extraction_output.claims
        log_info(f"Extracted {len(claims)} claims.")
        
        verified_claims = []
        
        # 2. Verification Loop
        log_info("Step 2: Verifying Claims...")
        for claim in claims:
            # 2a. Retrieval for this claim
            # We use a constructed query based on the claim
            retrieval_query = f"Verify claim: {claim.claim_text}"
            
            # Using defaults for retrieval params if not provided
            # Logic similar to run() but simplified for single claim
            retrieval_params = RetrievalParams_RetrievalAgent(
                top_k_vector=3,
                top_k_keyword=2, # focus on precision
                top_k_web=0,
                top_k_youtube=0
            )

            retrieval_input = RetrievalInput(
                user_id=user_id,
                session_id=session_id,
                rewritten_query=retrieval_query,
                retrievers_to_use=["vector", "keyword"],
                retrieval_params=retrieval_params,
                conversation_history=[],
                preferences=preferences if preferences else Preferences(response_style="concise", max_length=300, domain="general")
            )
            
            retrieval_output = await self.retriever_orchestrator.run(retrieval_input)
            chunks = [c.text for c in retrieval_output.chunks]
            
            # 2b. Verification
            verification_input = VerificationInput(
                claim_text=claim.claim_text,
                retrieved_chunks=chunks  # Passing text content
            )
            verification_output = await self.verifier.run(verification_input)
            
            verified_claims.append({
                "claim_text": claim.claim_text,
                "original_text": claim.original_text_segment,
                "status": verification_output.result.status,
                "confidence": verification_output.result.confidence_score,
                "explanation": verification_output.result.explanation,
                "evidence_chunks": retrieval_output.chunks # Store full chunk objects for frontend reference
            })
            
        # 3. Explanation Generation
        # 3. Explanation Generation
        log_info("Step 3: Generating Explanation...")
        try:
            explanation_output = await self.explainer.run(ExplanationInput(claims_with_verification=verified_claims))
            summary = explanation_output.summary
            detailed_report = explanation_output.detailed_report
        except Exception as e:
            log_error(f"Explanation failed: {e}")
            summary = "Could not generate summary."
            detailed_report = "Could not generate report."
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "verified_claims": verified_claims,
            "summary": summary,
            "detailed_report": detailed_report,
            "latency_ms": latency_ms
        }

