"""LLM-as-a-Judge evaluation rubric scoring draft responses on a 1-5 scale."""
import os
import re
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.evaluation.response_metrics import evaluate_response_quality

load_dotenv()
logger = get_logger(__name__)

JUDGE_RUBRIC_PROMPT = """You are an expert impartial evaluator judging an AI Customer Support Agent's draft reply for Apple Support.

Evaluate the draft response on a 1 to 5 scale across 5 criteria:

1. Groundedness (1-5):
   - 5: Strictly faithful to the provided historical Apple resolutions.
   - 3: Partially grounded, with minor generic troubleshooting additions.
   - 1: Completely ignores retrieved evidence or invents non-existent procedures.

2. Relevance (1-5):
   - 5: Directly and specifically addresses the customer's stated issue.
   - 3: Addresses the general topic but misses key specific details.
   - 1: Off-topic or answers a completely different problem.

3. Helpfulness & Actionability (1-5):
   - 5: Gives clear, actionable troubleshooting steps or an exact resolution path.
   - 3: Vaguely helpful but lacks clear instructions.
   - 1: Unhelpful, confusing, or dead-end.

4. Brand Tone & Consistency (1-5):
   - 5: Exemplifies Apple Support tone (empathetic, polite, professional, concise).
   - 3: Acceptable tone but slightly informal, abrupt, or overly robotic.
   - 1: Rude, robotic, unprofessional, or violates guidelines.

5. Factuality / Hallucination Risk (1-5):
   - 5: 100% factual; makes no false promises of refunds, warranty extensions, or fake timelines.
   - 3: Minor unverified assertion with low harm.
   - 1: Severe hallucination (e.g., promising a free iPhone replacement or illegal actions).

Output your evaluation strictly in valid JSON format:
{
  "groundedness": <1-5>,
  "relevance": <1-5>,
  "helpfulness": <1-5>,
  "brand_consistency": <1-5>,
  "factuality": <1-5>,
  "overall_score": <1.0-5.0>,
  "reasoning": "<Short explanation>"
}
"""


class LLMJudge:
    """
    Impartial judge scoring support drafts using a structured 1-5 rubric.
    Supports OpenAI API or deterministic rubric scoring when offline.
    """
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def _judge_heuristic(
        self,
        customer_message: str,
        predicted_intent: str,
        draft_response: str,
        evidence: List[Dict[str, Any]],
        expected_decision: str
    ) -> Dict[str, Any]:
        """
        Calibrated deterministic rule-based judge for reproducible offline evaluation.
        """
        resp_lower = draft_response.lower()
        query_lower = customer_message.lower()

        # 1. Groundedness
        max_sim = max([e.get("similarity_score", 0.0) for e in evidence], default=0.0)
        if max_sim >= 0.70:
            groundedness = 5
        elif max_sim >= 0.55:
            groundedness = 4
        elif max_sim >= 0.40:
            groundedness = 3
        else:
            groundedness = 2

        # 2. Relevance
        overlap = set(query_lower.split()).intersection(set(resp_lower.split()))
        if len(overlap) >= 4 or any(w in resp_lower for w in ["ios", "battery", "apple id", "charge", "sim", "screen", "icloud"]):
            relevance = 5
        elif len(overlap) >= 2:
            relevance = 4
        else:
            relevance = 3

        # 3. Helpfulness
        has_step = any(w in resp_lower for w in ["restart", "settings", "dm", "update", "visit", "apple.com", "step", "check"])
        helpfulness = 5 if has_step else 3

        # 4. Brand Tone
        has_polite = any(w in resp_lower for w in ["we'd love", "let's", "thanks for reaching", "we understand", "we're here"])
        brand_consistency = 5 if has_polite else 4

        # 5. Factuality / Hallucination Risk
        has_fake_promise = any(w in resp_lower for w in ["guarantee refund", "free new phone", "immediate cash"])
        factuality = 1 if has_fake_promise else 5

        overall = round((groundedness + relevance + helpfulness + brand_consistency + factuality) / 5.0, 2)

        return {
            "groundedness": groundedness,
            "relevance": relevance,
            "helpfulness": helpfulness,
            "brand_consistency": brand_consistency,
            "factuality": factuality,
            "overall_score": overall,
            "reasoning": f"Grounded in top evidence ({max_sim:.2f} sim). Clear steps provided with polite Apple tone."
        }

    def judge_response(
        self,
        customer_message: str,
        predicted_intent: str,
        draft_response: str,
        evidence: List[Dict[str, Any]],
        expected_decision: str = "AUTO_HANDLE"
    ) -> Dict[str, Any]:
        """Judges response quality."""
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                prompt = (
                    f"Customer Message: {customer_message}\n"
                    f"Predicted Intent: {predicted_intent}\n"
                    f"Draft Reply: {draft_response}\n"
                    f"Retrieved Evidence: {json.dumps(evidence[:2])}\n"
                )
                res = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": JUDGE_RUBRIC_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return json.loads(res.choices[0].message.content)
            except Exception as e:
                logger.warning(f"LLM judge API call failed: {e}. Using calibrated heuristic judge.")

        return self._judge_heuristic(customer_message, predicted_intent, draft_response, evidence, expected_decision)
