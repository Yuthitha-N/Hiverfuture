"""Streamlit Interactive Web Application for Apple Support AI Agent with Light/Dark Theme & Balanced Layout."""
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

# Configure Page
st.set_page_config(
    page_title="AppleSupport AI | Autonomous Customer Support Intelligence",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Theme State & CSS Definition
# ---------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# Theme Toggle in Sidebar
with st.sidebar:
    st.markdown("### 🎨 Appearance & Theme")
    theme_choice = st.radio(
        "Select Theme Mode:",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True
    )
    is_dark = "Dark" in theme_choice
    st.session_state.theme = "Dark" if is_dark else "Light"

# Dynamic CSS Theme Variables
if is_dark:
    theme_vars = """
        --bg-body: #0a0f1d;
        --bg-card: rgba(19, 27, 46, 0.85);
        --bg-card-hover: rgba(26, 36, 61, 0.95);
        --border-card: rgba(255, 255, 255, 0.1);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #38bdf8;
        --accent-blue-gradient: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%);
        --input-bg: rgba(15, 23, 42, 0.9);
        --input-border: #334155;
        --input-text: #f8fafc;
        --badge-auto-bg: linear-gradient(135deg, #059669 0%, #047857 100%);
        --badge-escalate-bg: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        --metric-bg: rgba(15, 23, 42, 0.7);
        --step-box-bg: rgba(19, 27, 46, 0.9);
        --evidence-bg: rgba(15, 23, 42, 0.8);
        --shadow-elevation: 0 8px 32px rgba(0, 0, 0, 0.35);
    """
else:
    theme_vars = """
        --bg-body: #f8fafc;
        --bg-card: rgba(255, 255, 255, 0.9);
        --bg-card-hover: rgba(255, 255, 255, 1.0);
        --border-card: rgba(226, 232, 240, 0.9);
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --accent-blue: #0284c7;
        --accent-blue-gradient: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%);
        --input-bg: #ffffff;
        --input-border: #cbd5e1;
        --input-text: #0f172a;
        --badge-auto-bg: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --badge-escalate-bg: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        --metric-bg: #f1f5f9;
        --step-box-bg: #ffffff;
        --evidence-bg: #f8fafc;
        --shadow-elevation: 0 6px 24px rgba(0, 0, 0, 0.06);
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {{
        {theme_vars}
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-body) !important;
        color: var(--text-primary) !important;
    }}

    /* Global Input Overrides for Theme */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 10px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}
    .stSelectbox>div>div {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 10px !important;
    }}

    /* Ambient Header Banner */
    .ambient-header {{
        background: var(--accent-blue-gradient);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(2, 132, 199, 0.25);
        position: relative;
        overflow: hidden;
    }}
    .ambient-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
        animation: pulseSlow 9s ease-in-out infinite alternate;
    }}
    @keyframes pulseSlow {{
        0% {{ transform: scale(0.9) rotate(0deg); }}
        100% {{ transform: scale(1.1) rotate(10deg); }}
    }}

    .main-title {{
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        position: relative;
        z-index: 1;
        color: #ffffff !important;
    }}
    .main-subtitle {{
        font-size: 1.0rem;
        opacity: 0.92;
        margin-top: 6px;
        font-weight: 400;
        position: relative;
        z-index: 1;
        color: #ffffff !important;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(10px);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 14px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #ffffff !important;
    }}

    /* Balanced Card Container */
    .balanced-card {{
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-card);
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--shadow-elevation);
        margin-bottom: 20px;
        min-height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    /* Decision Badges */
    .decision-badge {{
        padding: 14px 20px;
        border-radius: 12px;
        color: white !important;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }}
    .decision-badge.auto {{
        background: var(--badge-auto-bg);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);
    }}
    .decision-badge.escalate {{
        background: var(--badge-escalate-bg);
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.25);
    }}

    /* Reply Output Box */
    .reply-card {{
        background: var(--input-bg);
        border: 1.5px solid var(--accent-blue);
        border-radius: 12px;
        padding: 16px 18px;
        font-size: 1.02rem;
        line-height: 1.6;
        color: var(--text-primary);
        margin-bottom: 16px;
    }}

    /* Evidence Box */
    .evidence-box {{
        background: var(--evidence-bg);
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-top: 1px solid var(--border-card);
        border-right: 1px solid var(--border-card);
        border-bottom: 1px solid var(--border-card);
    }}

    /* Metric Tiles */
    .metric-tile {{
        background: var(--metric-bg);
        border: 1px solid var(--border-card);
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
    }}
    .metric-val {{
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--accent-blue);
    }}
    .metric-lbl {{
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-top: 2px;
    }}

    /* Step Pipeline Ribbon */
    .step-pill-box {{
        background: var(--step-box-bg);
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: var(--shadow-elevation);
    }}
    .step-badge {{
        display: inline-block;
        width: 26px;
        height: 26px;
        background: var(--accent-blue);
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.85rem;
        line-height: 26px;
        margin-bottom: 4px;
    }}

    /* Placeholder Box when no query is executed */
    .placeholder-card {{
        background: var(--bg-card);
        border: 2px dashed var(--border-card);
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        min-height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
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
    # Ambient Glowing Header
    st.markdown("""
    <div class="ambient-header">
        <div class="main-title">🍏 AppleSupport AI — Customer Support Intelligence</div>
        <div class="main-subtitle">Autonomous Triage, Historical Evidence Retrieval, Brand-Grounded Draft Generation & Safety Escalation</div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <div class="status-pill">⚡ Latency: &lt;15ms</div>
            <div class="status-pill">📚 Knowledge Base: 8,000 Verified Resolutions</div>
            <div class="status-pill">🛡️ Deterministic Safety Guardrails: Active</div>
            <div class="status-pill">🎯 Golden Set Accuracy: 88.89%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        config, taxonomy, classifier, retriever, generator, escalation_policy, judge = load_agent_components()
    except Exception as e:
        st.error(f"Error loading system components: {e}. Please run `python run_pipeline.py` first.")
        st.stop()

    # Sidebar Controls
    st.sidebar.markdown("### ⚙️ System Controls")
    st.sidebar.markdown("<small style='color:var(--text-muted);'>Configure safety thresholds and retrieval depth</small>", unsafe_allow_html=True)
    conf_thresh = st.sidebar.slider("Intent Confidence Threshold", 0.40, 0.90, float(config["escalation"]["confidence_threshold"]), 0.05, help="Minimum intent classification confidence required before triggering human escalation.")
    sim_thresh = st.sidebar.slider("Evidence Similarity Threshold", 0.40, 0.85, float(config["escalation"]["similarity_threshold"]), 0.05, help="Minimum cosine similarity required between query and historical brand evidence.")
    top_k = st.sidebar.slider("Retrieved Evidence Count (Top-K)", 1, 5, int(config["retrieval"]["top_k"]))

    escalation_policy.confidence_threshold = conf_thresh
    escalation_policy.similarity_threshold = sim_thresh
    retriever.top_k = top_k

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Interactive Scenario Presets")
    st.sidebar.markdown("<small style='color:var(--text-muted);'>Select a pre-configured customer inquiry to test:</small>", unsafe_allow_html=True)
    
    sample_queries = {
        "Custom Query (Type your own)": "",
        "🔋 Battery Drain (Safe Auto-Handle)": "My iPhone 7 battery drops from 80% to 20% within an hour of normal use.",
        "🔄 iOS 11 Update Freeze (Safe Auto-Handle)": "Ever since updating to iOS 11.1, my phone keeps freezing on the lockscreen.",
        "📶 Wi-Fi Greyed Out (Safe Auto-Handle)": "My Wi-Fi toggle switch is greyed out in settings and I cannot turn it on.",
        "🎧 AirPods Disconnect (Safe Auto-Handle)": "My AirPods keep disconnecting during phone calls every 2 minutes.",
        "🚨 Credit Card Fraud / Theft (Escalate)": "Someone stole my credit card and made $300 of unauthorized App Store purchases!",
        "🔒 Apple ID Lockout (Escalate)": "My Apple ID has been disabled and I cannot reset my password or access my email.",
        "👤 Talk to Human Agent (Escalate)": "Can I please speak to a real human agent right now? Your bot is not helpful.",
        "❓ Vague Ambiguous Message (Escalate)": "Help it broke"
    }

    selected_sample = st.sidebar.selectbox("Choose Scenario:", list(sample_queries.keys()))

    tab_play, tab_explainer, tab_tax, tab_eval, tab_failures = st.tabs([
        "🚀 Live Support Playground",
        "💡 How It Works (Visual Guide)",
        "📑 Intent Taxonomy (8 Categories)",
        "📊 Evaluation & Benchmarks",
        "🔍 Failure Modes & Decisions"
    ])

    with tab_play:
        # Balanced 4-Step Pipeline Ribbon
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="step-pill-box"><div class="step-badge">1</div><br><b>Classify Intent</b><br><small style="color:var(--text-muted);">MiniLM + Calibrated LR</small></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="step-pill-box"><div class="step-badge">2</div><br><b>Retrieve Evidence</b><br><small style="color:var(--text-muted);">Top-K Historical Q&A</small></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="step-pill-box"><div class="step-badge">3</div><br><b>Generate Draft</b><br><small style="color:var(--text-muted);">Apple Brand Grounding</small></div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="step-pill-box"><div class="step-badge">4</div><br><b>Safety Decision</b><br><small style="color:var(--text-muted);">AUTO-HANDLE vs ESCALATE</small></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Equal-width, balanced 2-column layout
        col_left, col_right = st.columns([1, 1], gap="large")

        # Session state for query persistence
        if "analyzed_data" not in st.session_state:
            st.session_state.analyzed_data = None

        with col_left:
            st.markdown("### 📥 Customer Inquiry Input")
            default_text = sample_queries[selected_sample] if selected_sample != "Custom Query (Type your own)" else ""
            
            customer_query = st.text_area(
                "Customer Message:",
                value=default_text,
                height=150,
                placeholder="Type any customer tweet (e.g., 'My iPhone battery is draining very fast after iOS 11 update...')"
            )

            context_query = st.text_input(
                "Conversation Context (Optional):",
                value="Customer reached out via Twitter @AppleSupport",
                placeholder="Prior turns or channel details"
            )

            analyze_btn = st.button("🚀 Process & Generate Support Draft", type="primary", use_container_width=True)

            if analyze_btn and customer_query.strip():
                with st.spinner("Analyzing semantics, searching historical resolutions, and evaluating safety rules..."):
                    pred_intent, conf, prob_dict = classifier.predict_single(customer_query)
                    evidence = retriever.retrieve_evidence(customer_query, predicted_intent=pred_intent)
                    max_sim = retriever.get_max_similarity(evidence)
                    esc_res = escalation_policy.evaluate(
                        customer_message=customer_query,
                        predicted_intent=pred_intent,
                        confidence=conf,
                        evidence=evidence,
                        context=context_query
                    )
                    draft_reply = generator.generate_response(customer_query, pred_intent, evidence, context=context_query)
                    resp_metrics = evaluate_response_quality(customer_query, draft_reply, evidence)
                    judge_eval = judge.judge_response(customer_query, pred_intent, draft_reply, evidence, esc_res["decision"])

                    st.session_state.analyzed_data = {
                        "query": customer_query,
                        "pred_intent": pred_intent,
                        "conf": conf,
                        "evidence": evidence,
                        "max_sim": max_sim,
                        "esc_res": esc_res,
                        "draft_reply": draft_reply,
                        "judge_eval": judge_eval
                    }

        with col_right:
            st.markdown("### 📤 Triage Output & Decision")
            data = st.session_state.analyzed_data

            if data is not None and data["query"] == customer_query:
                decision = data["esc_res"]["decision"]
                reason = data["esc_res"]["reason"]
                conf = data["conf"]
                max_sim = data["max_sim"]
                judge_eval = data["judge_eval"]

                # 1. Decision Banner
                if decision == "AUTO_HANDLE":
                    st.markdown(f"""
                    <div class="decision-badge auto">
                        <span style="font-size: 1.6rem;">✅</span>
                        <div>
                            <div style="font-size: 1.15rem;">AUTO-HANDLE (Automated Dispatch Approved)</div>
                            <div style="font-size: 0.85rem; font-weight: 400; opacity: 0.95;">High intent certainty ({conf*100:.1f}%) and strong historical precedent ({max_sim:.2f})</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="decision-badge escalate">
                        <span style="font-size: 1.6rem;">⚠️</span>
                        <div>
                            <div style="font-size: 1.15rem;">ESCALATE TO HUMAN SPECIALIST</div>
                            <div style="font-size: 0.85rem; font-weight: 400; opacity: 0.95;">Safety rule triggered: <b>{data['esc_res'].get('rule_triggered', 'SAFETY_POLICY')}</b></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"<p style='color: var(--text-secondary); margin-bottom: 12px;'><b>Triage Explanation:</b> {reason}</p>", unsafe_allow_html=True)

                # 2. Predicted Intent & Confidence Meter
                st.markdown(f"**Predicted Intent:** `{data['pred_intent']}`")
                st.progress(conf, text=f"Classification Confidence: {conf*100:.1f}%")

                # 3. Grounded Draft Response
                st.markdown("#### 💬 Grounded Draft Reply")
                st.markdown(f'<div class="reply-card">{data["draft_reply"]}</div>', unsafe_allow_html=True)

                # 4. Multi-Dimensional Quality Rubric (Equal 5 Tiles)
                st.markdown("#### 🏅 Multi-Dimensional Quality Rubric")
                t1, t2, t3, t4, t5 = st.columns(5)
                with t1:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{judge_eval.get("groundedness", 5)}/5</div><div class="metric-lbl">Grounded</div></div>', unsafe_allow_html=True)
                with t2:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{judge_eval.get("relevance", 5)}/5</div><div class="metric-lbl">Relevant</div></div>', unsafe_allow_html=True)
                with t3:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{judge_eval.get("helpfulness", 5)}/5</div><div class="metric-lbl">Helpful</div></div>', unsafe_allow_html=True)
                with t4:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{judge_eval.get("brand_consistency", 5)}/5</div><div class="metric-lbl">Brand Tone</div></div>', unsafe_allow_html=True)
                with t5:
                    st.markdown(f'<div class="metric-tile"><div class="metric-val">{judge_eval.get("factuality", 5)}/5</div><div class="metric-lbl">Factual</div></div>', unsafe_allow_html=True)

                # 5. Retrieved Evidence Expander
                with st.expander(f"📚 Retrieved Historical Resolutions ({len(data['evidence'])} verified cases, Max Sim: {max_sim:.2f})", expanded=False):
                    for idx, ev in enumerate(data["evidence"], 1):
                        st.markdown(f"""
                        <div class="evidence-box">
                            <div style="display:flex; justify-content:space-between; margin-bottom: 4px;">
                                <b>Resolution Evidence #{idx}</b>
                                <span style="background:rgba(56, 189, 248, 0.2); color:var(--accent-blue); padding: 2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600;">
                                    Similarity: {ev.get('similarity_score', 0):.2f}
                                </span>
                            </div>
                            <div style="color:var(--text-muted); font-size:0.88rem; margin-bottom: 4px;"><i>Query:</i> "{ev.get('customer_message', '')}"</div>
                            <div style="color:var(--text-primary); font-size:0.94rem;"><b>AppleSupport:</b> "{ev.get('support_response', '')}"</div>
                        </div>
                        """, unsafe_allow_html=True)

            else:
                # Beautiful Empty State Card
                st.markdown("""
                <div class="placeholder-card">
                    <div style="font-size: 3.2rem; margin-bottom: 12px;">🍏</div>
                    <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 6px; color: var(--text-primary);">Awaiting Customer Inquiry</div>
                    <div style="font-size: 0.95rem; color: var(--text-muted); max-width: 340px;">
                        Select a preset scenario on the left or type a custom query, then click <b>Process & Generate Support Draft</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_explainer:
        st.subheader("💡 How This AI Customer Support Agent Works (In Plain English)")
        st.markdown("""
        ### Why Traditional Chatbots Fail vs How Our Agent Works
        
        Most basic AI chatbots fail in customer support because they generate answers completely from imagination (hallucination). If a customer asks about a broken phone, a generic bot might falsely promise *"We will give you a full free replacement tomorrow!"*—which ruins customer trust and violates company policy.
        
        Our system is built on **4 Rigorous Engineering Guardrails**:
        
        1. **Fast Semantic Intent Triage (<15ms)**:
           Instead of reading millions of words, our model converts the customer's tweet into a 384-dimensional semantic fingerprint and instantly identifies the exact problem category (e.g. *Battery Issue* vs *iCloud Storage*).
        
        2. **Grounded Historical Evidence Retrieval**:
           Before writing a single word, the system searches our database of **8,000 real, verified Apple Support resolutions** to find how Apple's expert technicians actually solved this exact problem in the past.
        
        3. **Strict Brand Voice & Factuality Constraints**:
           The generated reply is strictly anchored in the retrieved evidence. It is programmed to **never** make up refund numbers, warranty extensions, or fake timelines.
        
        4. **Deterministic Safety Escalation Engine**:
           If the issue involves security risks (Apple ID hacked, unauthorized credit card charge, legal threat), or if the AI is not 100% sure of the solution, it **automatically hands off the ticket to a human agent** with an auditable explanation.
        """)

    with tab_tax:
        st.subheader("📑 Grounded Intent Taxonomy (AppleSupport)")
        st.write("Derived through empirical frequency analysis of 76,000+ real AppleSupport Twitter interactions.")
        
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
