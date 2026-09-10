# Decision Log: AI Customer Support Agent (AppleSupport)

This document records the **12 key non-obvious engineering and machine learning decisions** made during the design and implementation of the AI Customer Support Agent for **Hiver Future**.

---

### Decision 1: Brand Selection — AppleSupport over AmazonHelp or SpotifyCares
- **Decision**: Select `AppleSupport` as the single target brand from the Kaggle Customer Support on Twitter dataset.
- **Why**: AppleSupport contains 76,639 multi-turn conversations with over 95% clean English text and clear, technical troubleshooting issues (iOS updates, battery degradation, iCloud sync, AirPods bluetooth, hardware screen issues). Furthermore, it provides well-defined security boundaries (Apple ID lockouts, 2FA issues) that naturally test automated vs human escalation.
- **Alternatives Considered**: `AmazonHelp` (81k+ conversations, but ~40% non-English tweets in Spanish, Japanese, German, requiring extensive language filtering) and `SpotifyCares` (27k+ conversations, but narrower scope focused primarily on playlist/billing issues).
- **Trade-off**: Apple support queries often involve multi-symptom customer inquiries (e.g. battery drain + iOS update), requiring more robust intent handling.

---

### Decision 2: Domain-Grounded Intent Taxonomy of 8 Categories
- **Decision**: Define an empirical taxonomy of 8 mutually exclusive intents directly derived from historical Apple customer inquiries (`ios_software_update`, `battery_performance`, `apple_id_account_security`, `app_store_billing_subscriptions`, `connectivity_network_bluetooth`, `device_hardware_display`, `data_sync_backup_icloud`, `general_inquiry_features`).
- **Why**: 8 categories provide sufficient granularity to guide grounded retrieval without causing severe class fragmentation or excessive overlapping semantic boundaries.
- **Alternatives Considered**: A generic 4-class taxonomy (Too coarse to guide retrieval) or a 25-class granular taxonomy (Severe class imbalance and ambiguous decision boundaries).
- **Trade-off**: Inquiries spanning two domains (e.g. battery drain immediately following an iOS update) must be resolved to a single primary intent or handled as compound queries.

---

### Decision 3: Intent Classification Architecture — Sentence Transformers (`all-MiniLM-L6-v2`) + Calibrated Logistic Classifier
- **Decision**: Use 384-dimensional dense semantic embeddings from `all-MiniLM-L6-v2` with a balanced, regularized Logistic Classifier as the main classification engine.
- **Why**: Achieves inference latency <15ms on CPU, provides calibrated probability distributions for confidence thresholding, operates 100% offline, and captures semantic paraphrases far better than surface-level keyword matchers.
- **Alternatives Considered**: Large zero-shot LLM prompts (Expensive, high latency ~1000ms, non-deterministic confidence calibration) or pure TF-IDF (Fails on out-of-vocabulary synonyms and colloquial phrasing).
- **Trade-off**: Dense embeddings have slightly lower keyword precision on rare, exact model strings compared to n-gram TF-IDF on keyword-heavy training data.

---

### Decision 4: Class Imbalance Strategy — Class-Weighted Loss over SMOTE
- **Decision**: Employ balanced inverse-frequency class weighting (`class_weight='balanced'`) during classifier training rather than synthetic oversampling (SMOTE).
- **Why**: SMOTE in dense embedding space can generate unrealistic synthetic points along the boundary between disparate intents (e.g. synthesizing hybrid queries between billing and hardware). Inverse frequency weighting preserves true feature distributions while penalizing minority class misclassifications.
- **Alternatives Considered**: SMOTE / Random oversampling or downsampling the majority class (`ios_software_update`).
- **Trade-off**: Minority classes may exhibit slightly higher variance in confidence scores.

---

### Decision 5: Multi-Factor Explicit Escalation Policy vs LLM-Only Escalation
- **Decision**: Implement a deterministic, rule-based escalation engine evaluating classifier confidence (<0.65), retrieval similarity (<0.55), high-risk security keywords, and explicit human requests.
- **Why**: Customer safety, fraud prevention, and account security must not rely solely on LLM prompt compliance. An explicit rule engine provides guaranteed SLAs, auditable logs, and clear explanations for why a ticket was escalated.
- **Alternatives Considered**: Asking the LLM prompt to output `"ESCALATE"` at its discretion (Prone to prompt injection, hallucinations, and unpredictable edge cases).
- **Trade-off**: Hard thresholds may occasionally escalate emotionally charged customer tweets that could theoretically be resolved with standard advice.

---

### Decision 6: Semantic Vector Retrieval with Intent Pre-Filtering / Boosting
- **Decision**: Index historical AppleSupport resolutions using exact cosine similarity over normalized embeddings, with an intent-affinity boost for candidates matching the predicted intent.
- **Why**: Intent boosting prevents cross-domain topic bleeding (e.g. retrieving battery tips for a Wi-Fi issue) while still allowing strong semantic matches to surface if the customer's phrasing is nuanced.
- **Alternatives Considered**: Unfiltered global cosine similarity (Risks retrieving irrelevant resolutions with high surface lexical similarity).
- **Trade-off**: If intent classification fails, the retriever may slightly bias towards the incorrect intent category.

---

### Decision 7: Evidence-Grounded Response Synthesis with Hard Policy Boundaries
- **Decision**: Constrain the generator to troubleshooting actions explicitly present in retrieved historical evidence, forbidding fake compensation, warranty promises, or timeline estimates.
- **Why**: Customer trust is destroyed if an AI promises a free screen replacement or instant refund that Apple Retail will refuse. Strict grounding guarantees corporate compliance.
- **Alternatives Considered**: Unconstrained generative creativity for fluent conversational dialogue.
- **Trade-off**: When historical evidence is sparse, the agent offers standardized diagnostic steps and invites the customer to DM their iOS version rather than guessing a resolution.

---

### Decision 8: Curating a 180-Sample Golden Evaluation Set
- **Decision**: Hand-craft and stratify 180 representative evaluation examples across all 8 intents, edge cases, sensitive triggers, and ambiguous queries, strictly separated from the training corpus.
- **Why**: Real Twitter data contains noise and label leakage. A curated golden set with known ground-truth intents and expected handling decisions provides a rock-solid, leakage-free benchmark.
- **Alternatives Considered**: Splitting raw noisy Twitter data without manual validation (Tests noise rather than actual system competence).
- **Trade-off**: 180 curated examples require careful manual effort, but provide high-signal calibration.

---

### Decision 9: Structured 1–5 LLM-as-a-Judge Rubric with 5 Distinct Dimensions
- **Decision**: Evaluate response quality across 5 explicit dimensions: *Groundedness*, *Query Relevance*, *Helpfulness & Actionability*, *Brand Tone Consistency*, and *Factuality / Safety*.
- **Why**: Single composite scores (e.g. "Rate 1-10") conflate fluency with factual grounding. Multi-dimensional rubrics isolate whether an agent is polite but hallucinating vs grounded but abrupt.
- **Alternatives Considered**: BLEU / ROUGE only (Measure string overlap, not semantic helpfulness or customer support quality).
- **Trade-off**: Requires structured JSON parsing and rubric calibration against human judgements.

---

### Decision 10: Judge Calibration against Human Judgements
- **Decision**: Calibrate the automated judge on a 30-sample human-annotated benchmark, calculating Mean Absolute Error, agreement rate within 0.5 points, and analyzing specific disagreement cases.
- **Why**: Proves whether the automated evaluation harness is trustworthy and mirrors human expert standards.
- **Alternatives Considered**: Blindly trusting the LLM judge's headline numbers without empirical validation.
- **Trade-off**: Disagreements reveal slight judge biases (e.g. judge being stricter on generic DM requests than human evaluators).

---

### Decision 11: Subsampling Strategy — 10,000 Conversations Split 80/20
- **Decision**: Extract 10,000 clean, deduplicated AppleSupport conversation pairs (8,000 for training & vector knowledge base, 2,000 for validation).
- **Why**: Enables the entire pipeline to train, index, evaluate, and reproduce headline results in under 15 minutes on standard laptop CPUs without GPU clusters.
- **Alternatives Considered**: Processing all 76k Apple tweets (Unnecessary 45-minute runtime with diminishing returns on classification accuracy).
- **Trade-off**: Slightly smaller retrieval index (8,000 resolutions), though more than sufficient for high-density support topics.

---

### Decision 12: Dual Execution Mode (Full Local Offline & LLM API Integration)
- **Decision**: Architect the entire pipeline with a deterministic local offline grounding engine and calibrated judge, while supporting OpenAI/Gemini APIs via environment variables.
- **Why**: Allows any reviewer or interviewer to clone the repo, run `python run_pipeline.py`, and inspect all working results instantly without configuring paid API keys or exposing private credentials.
- **Alternatives Considered**: Requiring mandatory paid OpenAI API keys to run the pipeline.
- **Trade-off**: Requires maintaining both the prompt-based generator and the deterministic offline grounding synthesis engine.
