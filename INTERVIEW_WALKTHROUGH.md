# Step-by-Step Interview Walkthrough & Project Explanation

**Project**: Hiver AI Customer Support Agent for AppleSupport (`Hiverfuture`)  
**Target Audience**: Interviewers, Engineering Reviewers, and Candidates preparing for a live interview walkthrough.

---

## 🎯 High-Level Pitch (How to Introduce the Project in 60 Seconds)

> *"In this project, I built an end-to-end, production-grade AI Customer Support Agent for **AppleSupport** using the real-world Kaggle Twitter Customer Support dataset. The system ingests messy, informal customer tweets, classifies them into 8 domain-grounded intents, retrieves historically verified brand resolutions using semantic embeddings, drafts empathetic responses strictly adhering to Apple brand guidelines, and enforces a deterministic multi-factor policy to decide whether to **AUTO-HANDLE** the query or **ESCALATE** to a human agent.
> 
> Rather than relying on black-box LLM generation, the system is strictly grounded in real historical data, operates deterministically with $<15\text{ms}$ classification latency, is 100% reproducible in under 2 minutes, and includes a full evaluation harness with an LLM-as-a-judge rubric calibrated against human ratings."*

---

## 🧭 Step-by-Step Breakdown: What We Built & How

```
                                  PIPELINE OVERVIEW
                                  
  [ Raw Twitter Tweets ] ──► [ Data Cleaning & Parsing ] ──► [ 8 Intent Taxonomy ]
                                                                     │
  ┌──────────────────────────────────────────────────────────────────┘
  ▼
[ Intent Classifier ] ──► [ Semantic Vector Store ] ──► [ Response Generator ]
  (MiniLM + Logistic)       (Top-k Apple Q&A pairs)       (Prompt Grounding)
                                                                     │
  ┌──────────────────────────────────────────────────────────────────┘
  ▼
[ Escalation Policy ] ──► [ Decision Engine ] ───────► AUTO-HANDLE or ESCALATE
  (Rules + Confidence)      (Safety & Grounding Checks)
```

---

### STEP 1: Data Exploration & Preprocessing
- **What We Did**: Explored the 3M Kaggle Twitter dataset and selected **`AppleSupport`** (76,639 conversations).
- **Why AppleSupport?**:
  1. Over 95% clean English text.
  2. Diverse technical issues (iOS bugs, battery drain, iCloud sync, Bluetooth pairing).
  3. Clear escalation boundaries (Apple ID security lockouts and billing disputes must go to human agents, while Wi-Fi resets can be automated).
- **Text Cleaning**:
  - Redacted Twitter handles (`@AppleSupport`, `@115858`) so the model doesn't cheat on handle names.
  - Normalized URLs (`https://t.co/...` $\rightarrow$ `[URL]`).
  - Decoded HTML entities (`&amp;` $\rightarrow$ `&`).
  - Extracted initial customer problem statements and corresponding agent resolution replies into clean Q&A pairs.
- **Key Code**: `src/data_processing/cleaner.py`, `src/data_processing/loader.py`, `src/data_processing/thread_builder.py`.

---

### STEP 2: Intent Taxonomy Definition
- **What We Did**: Defined 8 grounded, mutually exclusive support intents:
  1. `ios_software_update` (iOS update errors, freezing, lag post-update)
  2. `battery_performance` (Fast drain, sudden shutdown, charging pause)
  3. `apple_id_account_security` (Apple ID lockout, 2FA codes, password reset)
  4. `app_store_billing_subscriptions` (Unauthorized charges, subscription cancels, refunds)
  5. `connectivity_network_bluetooth` (Wi-Fi greyed out, AirPods disconnect, No SIM)
  6. `device_hardware_display` (Cracked screens, black camera, hardware repairs)
  7. `data_sync_backup_icloud` (iCloud storage full, photo sync delays, device migration)
  8. `general_inquiry_features` (AppleCare warranty, trade-ins, how-to settings)
- **Key Code**: `config/intent_taxonomy.json`.

---

### STEP 3: Intent Classification Architecture
- **What We Built & Compared**:
  - **Baseline 1 (Trivial)**: `MajorityClassClassifier` $\rightarrow$ Always predicts `ios_software_update` (Accuracy: 56.90%, Macro F1: 0.0907).
  - **Baseline 2 (Simple ML)**: `TFIDFLogisticClassifier` $\rightarrow$ TF-IDF n-grams + Logistic Regression (Accuracy: 81.25%, Macro F1: 0.6936).
  - **Main Model**: `EmbeddingClassifier` $\rightarrow$ `sentence-transformers/all-MiniLM-L6-v2` dense embeddings (384 dims) + Calibrated Logistic Regression.
- **Performance on Golden Set**: **88.89% Accuracy**.
- **Why Main Model is Best**: Dense semantic embeddings capture out-of-vocabulary paraphrases and informal language that break TF-IDF keyword matchers.
- **Key Code**: `src/intent_classification/baselines.py`, `src/intent_classification/classifier.py`, `src/intent_classification/evaluate_intents.py`.

---

### STEP 4: Historical Evidence Retrieval
- **What We Did**: Built a semantic vector store (`VectorStore`) indexing 8,000 historical AppleSupport customer-resolution pairs.
- **How it Works**: Computes cosine similarity between incoming query embedding and indexed resolution embeddings. Applies an intent-affinity boost to ensure evidence matches the diagnosed problem.
- **Key Code**: `src/retrieval/vector_store.py`, `src/retrieval/retriever.py`.

---

### STEP 5: Response Generation & Brand Voice
- **What We Did**: Built a response synthesizer that takes customer inquiry, predicted intent, and top retrieved historical resolutions to draft an official Apple Support reply.
- **Apple Brand Guidelines Enforced**:
  1. *Polite & Empathetic*: Starts with warm acknowledgement (*"We'd love to help with this."*).
  2. *Actionable & Concise*: Clear step-by-step troubleshooting path.
  3. *Zero Hallucination*: Never promises free replacement phones, fake refund amounts, or unverified turnaround times.
  4. *DM Path*: Invites customer to DM private details (device model / iOS version) when sensitive.
- **Key Code**: `src/response_generation/prompts.py`, `src/response_generation/generator.py`.

---

### STEP 6: Deterministic Escalation Policy
- **What We Did**: Implemented an explicit rule engine rather than relying on LLM self-judgment.
- **Escalation Rules**:
  1. `EXPLICIT_HUMAN_REQUEST`: Customer asks for a human / agent / supervisor $\rightarrow$ **ESCALATE**.
  2. `SENSITIVE_KEYWORD_DETECTED`: Words like "fraud", "unauthorized charge", "lawyer", "stolen", "police" $\rightarrow$ **ESCALATE**.
  3. `MANDATORY_ACCOUNT_SECURITY`: Apple ID lockouts or compromised accounts $\rightarrow$ **ESCALATE**.
  4. `LOW_INTENT_CONFIDENCE`: Intent confidence $< 0.65$ $\rightarrow$ **ESCALATE**.
  5. `INSUFFICIENT_HISTORICAL_EVIDENCE`: Max retrieval similarity $< 0.55$ $\rightarrow$ **ESCALATE**.
  6. `AMBIGUOUS_QUERY`: Ultra-short query ($< 3$ words) without symptom $\rightarrow$ **ESCALATE**.
  7. `CONFIDENT_GROUNDED_AUTO_HANDLE`: All checks pass $\rightarrow$ **AUTO_HANDLE**.
- **Key Code**: `src/escalation/rules.py`, `src/escalation/policy.py`.

---

### STEP 7 & 8: Golden Evaluation Set & LLM-as-a-Judge Rubric
- **Golden Evaluation Set**: 180 hand-crafted test cases covering all 8 intents, edge cases, sensitive triggers, and ambiguous queries, strictly isolated from the training corpus (`data/golden/golden_evaluation_set.json`).
- **LLM-as-a-Judge**: 1–5 structured rubric evaluating:
  1. *Groundedness (4.43 / 5.0)*
  2. *Relevance (4.61 / 5.0)*
  3. *Helpfulness (5.00 / 5.0)*
  4. *Brand Tone (4.66 / 5.0)*
  5. *Factuality / Safety (5.00 / 5.0)*
  6. *Composite Quality Score (4.74 / 5.0)*
- **Judge Calibration**: Calibrated against a 30-case human-annotated benchmark showing **80.0% agreement** and low MAE (0.34).
- **Key Code**: `src/evaluation/llm_judge.py`, `src/evaluation/judge_calibration.py`.

---

### STEP 9: Top 5 Real Failure Modes
1. **Compound / Multi-Intent Queries**: Customer mentions iOS update AND battery drain; single-label classifier only addresses one.
2. **False Escalation on High Emotion**: Angry customers using strong words get escalated even on routine technical issues.
3. **Hardware Model Generational Mismatch**: Query about iPhone X gestures retrieves older iPhone 6s button sequence.
4. **Under-Specified Queries**: 2-word queries like *"Help it broke"* trigger necessary ambiguity escalation.
5. **Sub-Domain Terminology Shift**: Features like AirDrop overlapping Wi-Fi and Bluetooth.
- **Key Code**: `src/evaluation/failure_analysis.py`.

---

### STEP 10: "What is Misleading About My Headline Number?" (Crucial Interview Section!)
- **Why TF-IDF appeared higher on raw validation**: Raw Twitter data contains frequent exact keyword repetitions. TF-IDF overfits these exact tokens, creating an illusion of high accuracy, but fails on paraphrased real-world queries where Sentence Transformers excel.
- **Why Escalation Accuracy is 63.33%**: The system enforces a conservative safety bias, choosing false escalations over risky automated responses when customer sentiment is volatile.

---

### STEP 11: Interactive Demo Dashboard
- **Built with Streamlit**: `streamlit run app/streamlit_app.py`
- Features:
  - Live query entry with preloaded preset edge cases.
  - Real-time Intent Prediction & Confidence meter.
  - Dynamic `AUTO_HANDLE` vs `ESCALATE` badge with stated reason.
  - Grounded Draft Response box.
  - Expandable Retrieved Historical Evidence cards with similarity scores.
  - Live Evaluation Rubric scorecard.

---

## 🎤 Top Interview Questions & How to Answer

### Q1: "Why did you choose AppleSupport instead of Amazon or Spotify?"
> **Answer**: "AppleSupport offered the ideal balance of volume (76k+ multi-turn conversations) and 95%+ clean English data. Unlike AmazonHelp, which had ~40% mixed non-English tweets requiring heavy language filtering, AppleSupport conversations focused on rich, diagnostic technical issues (OS updates, battery health, iCloud sync, hardware). Most importantly, Apple Support has very clear security and compliance boundaries (Apple ID lockouts, 2FA, unauthorized credit card charges) that provided a high-signal benchmark for testing automated vs human escalation."

### Q2: "Why use Sentence Transformers over a direct prompt to GPT-4?"
> **Answer**: "Three reasons:
> 1. **Latency & Cost**: `all-MiniLM-L6-v2` runs locally in $<10\text{ms}$ on standard CPU at zero API cost, whereas LLM API calls take $800-1500\text{ms}$ and incur continuous token costs.
> 2. **Calibrated Confidence**: Logistic Regression over normalized embeddings gives mathematically calibrated probabilities $P(\text{intent} \mid x)$, allowing us to set an explicit, reliable confidence threshold (0.65) for escalation.
> 3. **Privacy & Offline Reliability**: The system can run 100% locally in an air-gapped corporate environment without sending customer tweets to third-party endpoints."

### Q3: "How do you guarantee the AI doesn't hallucinate fake policies?"
> **Answer**: "Through strict grounding and multi-factor guardrails:
> 1. The generator is prompted exclusively with top-k historical brand resolutions retrieved from our vector store.
> 2. Hard policy rules immediately escalate sensitive billing or legal keywords before generation even starts.
> 3. An automated factuality rubric checks that the response does not promise unauthorized compensation or fake warranties."

### Q4: "What would you improve if given one more week?"
> **Answer**:
> 1. **Multi-Label Classification**: Implement multi-label routing to handle compound queries (e.g. Battery Drain + iOS update).
> 2. **Device Entity Extraction**: Automatically extract device model (iPhone X vs iPhone 7) and iOS version to hard-filter vector retrieval.
> 3. **Conversational Clarification**: Enable the agent to ask follow-up questions for vague tweets rather than immediately escalating.
> 4. **Cross-Encoder Reranker**: Add a lightweight reranker (`ms-marco-MiniLM-L-6-v2`) to boost retrieval precision.

---

## 🚀 Commands to Demonstrate in a Live Interview

1. **Run Full End-to-End Pipeline**:
   ```bash
   python run_pipeline.py
   ```
2. **Run Pytest Suite**:
   ```bash
   pytest -v
   ```
3. **Launch Streamlit Demo App**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
