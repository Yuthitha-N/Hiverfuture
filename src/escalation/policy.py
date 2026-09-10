"""Deterministic Escalation Policy Engine."""
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.escalation.rules import check_sensitive_keywords, check_human_agent_request, is_ambiguous_or_unclear

logger = get_logger(__name__)


class EscalationPolicy:
    """
    Transparent, rule-based escalation policy engine.
    Evaluates incoming query, predicted intent, confidence score, and retrieved evidence
    to determine whether the ticket should be AUTO_HANDLED or ESCALATED_TO_HUMAN.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_yaml_config(config_path)
        esc_cfg = self.config.get("escalation", {})
        self.confidence_threshold = esc_cfg.get("confidence_threshold", 0.65)
        self.similarity_threshold = esc_cfg.get("similarity_threshold", 0.55)
        self.sensitive_keywords = esc_cfg.get("sensitive_keywords", [])
        self.human_triggers = esc_cfg.get("human_agent_triggers", [])

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        evidence: List[Dict[str, Any]],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluates support interaction against explicit escalation rules.
        Returns:
            {
                "decision": "AUTO_HANDLE" or "ESCALATE",
                "reason": "Detailed human-readable explanation",
                "rule_triggered": "Rule identifier",
                "metrics": {...}
            }
        """
        full_text = f"{customer_message} {context}".strip()
        max_sim = max([item.get("similarity_score", 0.0) for item in evidence], default=0.0)

        # 1. Explicit Human Agent Request
        is_human_req, trigger = check_human_agent_request(full_text, self.human_triggers)
        if is_human_req:
            return {
                "decision": "ESCALATE",
                "reason": f"Customer explicitly requested a human representative (trigger: '{trigger}').",
                "rule_triggered": "EXPLICIT_HUMAN_REQUEST",
                "metrics": {"confidence": confidence, "max_similarity": max_sim}
            }

        # 2. Sensitive / High-Risk Keyword Trigger
        is_sensitive, kw = check_sensitive_keywords(full_text, self.sensitive_keywords)
        if is_sensitive:
            return {
                "decision": "ESCALATE",
                "reason": f"Detected high-risk/sensitive keyword '{kw}' requiring manual verification and protocol compliance.",
                "rule_triggered": "SENSITIVE_KEYWORD_DETECTED",
                "metrics": {"confidence": confidence, "max_similarity": max_sim}
            }

        # 3. Intent-Specific Mandatory Escalation (Apple ID Security / Unauthorized Billing)
        if predicted_intent == "apple_id_account_security" and any(w in full_text.lower() for w in ["locked", "disabled", "hacked", "stolen", "unauthorized"]):
            return {
                "decision": "ESCALATE",
                "reason": "Apple ID account security lockout or compromise requires identity verification by a certified specialist.",
                "rule_triggered": "MANDATORY_ACCOUNT_SECURITY",
                "metrics": {"confidence": confidence, "max_similarity": max_sim}
            }

        # 4. Low Classifier Confidence
        if confidence < self.confidence_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Intent classification confidence ({confidence:.2f}) is below acceptable safety threshold ({self.confidence_threshold:.2f}).",
                "rule_triggered": "LOW_INTENT_CONFIDENCE",
                "metrics": {"confidence": confidence, "max_similarity": max_sim}
            }

        # 5. Low Retrieval Similarity (Insufficient Historical Precedent)
        if max_sim < self.similarity_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Historical resolution similarity ({max_sim:.2f}) is below grounding threshold ({self.similarity_threshold:.2f}); insufficient precedent to auto-respond safely.",
                "rule_triggered": "INSUFFICIENT_HISTORICAL_EVIDENCE",
                "metrics": {"confidence": confidence, "max_similarity": max_sim}
            }

        # 6. Ambiguous / Context-Deficient Query
        if is_ambiguous_or_unclear(customer_message, min_words=3):
            return {
                "decision": "ESCALATE",
                "reason": "Customer inquiry is too ambiguous or brief to diagnose safely without human probing.",
                "rule_triggered": "AMBIGUOUS_QUERY",
                "metrics": {"confidence": confidence, "max_similarity": max_sim}
            }

        # 7. Passed All Safety Checks -> AUTO_HANDLE
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"High intent confidence ({confidence:.2f}) and strong historical precedent ({max_sim:.2f}) allow safe automated response.",
            "rule_triggered": "CONFIDENT_GROUNDED_AUTO_HANDLE",
            "metrics": {"confidence": confidence, "max_similarity": max_sim}
        }
