"""Streamlit Interactive Web Application for Hiver AI Customer Support Agent."""
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import get_project_root, load_yaml_config, load_intent_taxonomy
from src.intent_classification.classifier import EmbeddingClassifier
from src.retrieval.retriever import HistoricalRetriever
from src.response_generation.generator import ResponseGenerator
from src.escalation.policy import EscalationPolicy
from src.evaluation.response_metrics import evaluate_response_quality
from src.evaluation.llm_judge import LLMJudge

st.set_page_config(
    page_title="Apple Support AI Agent | Hiver Take-Home",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1d1d1f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #86868b;
        margin-bottom: 1.5rem;
    }
    .badge-auto {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #a5d6a7;
    }
    .badge-escalate {
        background-color: #ffebee;
        color: #c62828;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #ef9a9a;
    }
    .evidence-card {
        background-color: #f8f9fa;
        border-left: 4px solid #0071e3;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .reply-box {
        background-color: #f5f5f7;
        border: 1px solid #d2d2d7;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.05rem;
        line-height: 1.5;
        color: #1d1d1f;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent_components():
    root = get_project_root()
    config = load_yaml_config()
    taxonomy = load_intent_taxonomy()

    classifier_path = root / config["classification"]["saved_model_path"]
    classifier = EmbeddingClassifier.load(classifier_path)
    retriever = HistoricalRetriever()
    generator = ResponseGenerator()
    escalation_policy = EscalationPolicy()
    judge = LLMJudge()

    return config, taxonomy, classifier, retriever, generator, escalation_policy, judge


def main():
    st.markdown('<div class="main-header">🍏 AppleSupport AI Customer Support Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hiver SDE Intern Take-Home Project | Autonomous Triage, Evidence Retrieval & Grounded Generation</div>', unsafe_allow_html=True)

    try:
        config, taxonomy, classifier, retriever, generator, escalation_policy, judge = load_agent_components()
    except Exception as e:
        st.error(f"Error loading system components: {e}. Please run `python run_pipeline.py` first to train models and build indices.")
        st.stop()

    # Sidebar Controls & Presets
    st.sidebar.header("⚙️ Agent Settings")
    conf_thresh = st.sidebar.slider("Intent Confidence Threshold", 0.40, 0.90, float(config["escalation"]["confidence_threshold"]), 0.05)
    sim_thresh = st.sidebar.slider("Retrieval Similarity Threshold", 0.40, 0.85, float(config["escalation"]["similarity_threshold"]), 0.05)
    top_k = st.sidebar.slider("Retrieved Evidence Count (Top-K)", 1, 5, int(config["retrieval"]["top_k"]))

    escalation_policy.confidence_threshold = conf_thresh
    escalation_policy.similarity_threshold = sim_thresh
    retriever.top_k = top_k

    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Sample Test Inquiries")
    sample_queries = {
        "Custom Query": "",
        "Battery Drain (Auto-Handle)": "My iPhone 7 battery drops from 80% to 20% within an hour of normal use.",
        "iOS Update Bug (Auto-Handle)": "Ever since updating to iOS 11.1, my phone keeps freezing on the lockscreen.",
        "Wi-Fi Connection (Auto-Handle)": "My Wi-Fi toggle switch is greyed out in settings and I cannot turn it on.",
        "AirPods Disconnect (Auto-Handle)": "My AirPods keep disconnecting during calls every 2 minutes.",
        "Stolen / Fraud (Escalate)": "Someone stole my credit card and made $300 of unauthorized App Store purchases!",
        "Apple ID Lockout (Escalate)": "My Apple ID has been disabled and I cannot reset my password or access my email.",
        "Human Request (Escalate)": "Can I please speak to a real human agent right now? Your bot is not helpful.",
        "Ambiguous Message (Escalate)": "Help it broke"
    }

    selected_sample = st.sidebar.selectbox("Choose a preset query:", list(sample_queries.keys()))

    tab_play, tab_tax, tab_eval, tab_failures = st.tabs([
        "🚀 Live Agent Playground",
        "📑 Intent Taxonomy",
        "📊 Evaluation & Benchmarks",
        "🔍 Failure Modes & Decisions"
    ])

    with tab_play:
        col_in, col_out = st.columns([1, 1], gap="medium")

        with col_in:
            st.subheader("📥 Incoming Customer Tweet")
            default_text = sample_queries[selected_sample] if selected_sample != "Custom Query" else ""
            customer_query = st.text_area(
                "Customer Message:",
                value=default_text,
                height=140,
                placeholder="Type a customer tweet (e.g., 'My iPhone battery is draining very fast after iOS 11 update...')"
            )

            context_query = st.text_input(
                "Conversation Context (optional):",
                value="Customer reached out via Twitter @AppleSupport",
                placeholder="Prior turns or metadata"
            )

            analyze_btn = st.button("🚀 Process & Generate Support Draft", type="primary", use_container_width=True)

        with col_out:
            st.subheader("📤 Agent Output & Triage Decision")
            if analyze_btn and customer_query.strip():
                with st.spinner("Classifying intent, retrieving historical evidence, and drafting grounded reply..."):
                    # 1. Intent Classification
                    pred_intent, conf, prob_dict = classifier.predict_single(customer_query)

                    # 2. Vector Retrieval
                    evidence = retriever.retrieve_evidence(customer_query, predicted_intent=pred_intent)
                    max_sim = retriever.get_max_similarity(evidence)

                    # 3. Escalation Decision
                    esc_res = escalation_policy.evaluate(
                        customer_message=customer_query,
                        predicted_intent=pred_intent,
                        confidence=conf,
                        evidence=evidence,
                        context=context_query
                    )
                    decision = esc_res["decision"]
                    reason = esc_res["reason"]

                    # 4. Draft Generation
                    draft_reply = generator.generate_response(customer_query, pred_intent, evidence, context=context_query)

                    # 5. Quality & Rubric Evaluation
                    resp_metrics = evaluate_response_quality(customer_query, draft_reply, evidence)
                    judge_eval = judge.judge_response(customer_query, pred_intent, draft_reply, evidence, decision)

                # Render Decision Badge
                if decision == "AUTO_HANDLE":
                    st.markdown(f'<div class="badge-auto">✅ AUTO-HANDLE ({conf*100:.1f}% confidence)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-escalate">⚠️ ESCALATE TO HUMAN ({esc_res.get("rule_triggered", "POLICY")})</div>', unsafe_allow_html=True)

                st.markdown(f"**Triage Reason:** {reason}")

                # Predicted Intent with Confidence Bar
                st.markdown(f"**Predicted Intent:** `{pred_intent}`")
                st.progress(conf, text=f"Classification Confidence: {conf*100:.1f}%")

                # Draft Response
                st.markdown("##### 💬 Grounded Draft Reply")
                st.markdown(f'<div class="reply-box">{draft_reply}</div>', unsafe_allow_html=True)

                # Quality Scorecard
                st.markdown("##### 🏅 Evaluation Rubric Score")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Groundedness", f"{judge_eval.get('groundedness', 5)}/5")
                m2.metric("Relevance", f"{judge_eval.get('relevance', 5)}/5")
                m3.metric("Helpfulness", f"{judge_eval.get('helpfulness', 5)}/5")
                m4.metric("Brand Tone", f"{judge_eval.get('brand_consistency', 5)}/5")

                # Retrieved Evidence Expander
                with st.expander(f"📚 Retrieved Historical Brand Evidence ({len(evidence)} examples, Max Sim: {max_sim:.2f})", expanded=False):
                    for idx, ev in enumerate(evidence, 1):
                        st.markdown(f"""
                        <div class="evidence-card">
                            <b>Historical Query #{idx} (Similarity: {ev.get('similarity_score', 0):.2f})</b><br>
                            <i>"{ev.get('customer_message', '')}"</i><br><br>
                            <b>Official AppleSupport Reply:</b><br>
                            "{ev.get('support_response', '')}"
                        </div>
                        """, unsafe_allow_html=True)

            elif analyze_btn:
                st.warning("Please enter a customer message to analyze.")

    with tab_tax:
        st.subheader("📑 Grounded Intent Taxonomy (AppleSupport)")
        st.write("Derived through empirical clustering and frequency analysis of 76,000+ AppleSupport Twitter interactions.")
        
        tax_data = []
        for item in taxonomy["intents"]:
            tax_data.append({
                "Intent Key": f"`{item['intent']}`",
                "Description": item["description"],
                "Sample Trigger Keywords": ", ".join(item.get("keywords", [])[:5]),
                "Historical Freq (%)": f"{item.get('historical_frequency_pct', 0.0)}%"
            })
        st.table(pd.DataFrame(tax_data))

    with tab_eval:
        st.subheader("📊 Model & Pipeline Evaluation Summary")
        root = get_project_root()
        metrics_file = root / "outputs/metrics/intent_classification_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            summary_rows = []
            for k, v in metrics_data.items():
                summary_rows.append({
                    "Model": v["model_name"],
                    "Accuracy": f"{v['accuracy']*100:.2f}%",
                    "Macro F1": f"{v['macro_f1']:.4f}",
                    "Weighted F1": f"{v['weighted_f1']:.4f}"
                })
            st.table(pd.DataFrame(summary_rows))

        golden_eval_file = root / "outputs/evaluations/golden_set_evaluation_results.json"
        if golden_eval_file.exists():
            with open(golden_eval_file, "r", encoding="utf-8") as f:
                gold_records = json.load(f)

            st.markdown(f"##### 🎯 Golden Set Evaluation ({len(gold_records)} Curated Hand-Labelled Cases)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Intent Accuracy", f"{np.mean([r['intent_correct'] for r in gold_records])*100:.1f}%")
            c2.metric("Escalation Decision Accuracy", f"{np.mean([r['decision_correct'] for r in gold_records])*100:.1f}%")
            c3.metric("Avg LLM Judge Score", f"{np.mean([r['judge_evaluation']['overall_score'] for r in gold_records]):.2f} / 5.0")
            c4.metric("Avg Groundedness", f"{np.mean([r['judge_evaluation']['groundedness'] for r in gold_records]):.2f} / 5.0")

    with tab_failures:
        st.subheader("🔍 Top 5 Real Failure Modes & Root Causes")
        failures_file = get_project_root() / "outputs/metrics/top_5_failure_modes.json"
        if failures_file.exists():
            with open(failures_file, "r", encoding="utf-8") as f:
                fail_data = json.load(f)
            for item in fail_data.get("top_5_failure_modes", []):
                with st.expander(f"#{item['rank']} {item['category']}", expanded=True):
                    st.markdown(f"**Description:** {item['description']}")
                    st.markdown(f"**Real Example:** `{item['real_example']}`")
                    st.markdown(f"**Expected Behavior:** {item['expected_behavior']}")
                    st.markdown(f"**Actual Behavior:** {item['actual_behavior']}")
                    st.markdown(f"**Likely Cause:** {item['likely_cause']}")
                    st.markdown(f"**Mitigation / Future Work:** {item['possible_improvement']}")


if __name__ == "__main__":
    main()
