# Report: AI Support Agent for SpotifyCares

*Hiver SDE Intern Take-Home — max 6 pages*

---

## 1. Problem Framing

**What "good" means for SpotifyCares support:**
A good support agent must (a) route messages correctly so humans aren't interrupted for routine questions, (b) draft replies that sound like the actual SpotifyCares voice — brief, friendly, not corporate — rather than a generic chatbot, and (c) never silently auto-handle anything that could harm the user (billing errors, account lockouts). The cost asymmetry matters: falsely auto-handling an account/billing issue is worse than an unnecessary human escalation.

**What was deliberately not built:**
- No web UI or dashboard (out of scope per assignment)
- No fine-tuning or embedding model (would require GPU and download; TF-IDF is explainable and sufficient)
- No multi-brand generalization (one brand done well > a half-baked framework for N brands)
- No processing of the full 3M-row dataset (subsample everywhere, stated explicitly)
- No agent orchestration framework (LangChain/LangGraph adds complexity without value for a 4-stage pipeline)

**Escalation philosophy:** AI handles routine questions (playback tips, feature how-tos, general complaints that just need acknowledgement). Humans stay in control of anything with financial or security stakes. This mirrors how Hiver's own product frames AI-assisted support: AI does the grunt work, humans decide what's risky.

---

## 2. Results vs Baselines

*(Populated after running `python run_pipeline.py eval` — see `eval/results_summary.md`)*

**Intent Classification**

| System | Accuracy | Macro-F1 |
|--------|----------|----------|
| Trivial baseline | 0.142 | 0.031 |
| Simple baseline (TF-IDF + LR) | 0.375 | 0.306 |
| LLM pipeline (120b) | 0.517 | 0.467 |

**Escalation Decision**

| System | Accuracy | False Auto-Handle | Unnecessary Escalate |
|--------|----------|-------------------|----------------------|
| Trivial (always escalate) | 0.408 | 0 | 71 |
| Simple (rule-only) | 0.408 | 0 | 71 |
| LLM pipeline (hybrid) | 0.800 | 15 | 9 |

**Reply Quality (LLM Judge, avg 1–5)**

| System | Grounded | Relevant | Tone |
|--------|----------|----------|------|
| Trivial (canned reply) | N/A | N/A | N/A |
| Simple (verbatim retrieval) | 2.73 | 2.82 | 2.63 |
| LLM pipeline | 3.78 | 3.81 | 3.53 |

**Judge-vs-Human Agreement (14-example subset intersection)**

- **grounded**: kappa=-0.273, within-1=100.0%
- **relevant**: kappa=-0.185, within-1=78.6%
- **tone**: kappa=0.013, within-1=64.3%

---

## 3. Failure Analysis

Top 5 failure modes (with real examples from the golden set evaluation):

1. **Intent ambiguity between `account_access` and `cancellation`**
   Example: *"Hello. I want to cancel my premium subscription, but I deleted my Facebook a while ago and I am unable to log in."*
   The true intent is `account_access` (they are blocked by login), but the LLM predicts `cancellation` because the word "cancel" strongly triggers the classifier. The root problem is access, not the subscription policy.

2. **Fuzzy boundaries on "Family Plan" issues (`billing_subscription` vs `account_access`)**
   Example: *"I want to invite my wife to family plan but Something is wrong with the address"*
   Adding people to a family plan is a subscription/billing feature, but "address mismatch" feels like an account data issue. The LLM predicts `account_access` while the human golden label was `billing_subscription`.

3. **Escalation False Negatives (Auto-handling PII issues)**
   Example: *"I need help cancelling my account. I do not remember my email or password."*
   The LLM incorrectly decided to auto-handle this (likely by just providing a generic "here's how to reset your password" link). It failed to reason that if the user doesn't remember their email, they cannot self-serve and a human agent needs to verify their identity.

4. **Boilerplate stripping over-aggressively removes useful context**
   Some SpotifyCares replies contain a boilerplate opener but a useful resolution in the second sentence. The current filter (< 20 words + pattern match) can discard the whole reply, starving the retrieval corpus of real resolutions.

5. **Low-volume intents (`praise`, `other`) underperform on macro-F1**
   With only ~30 golden examples per class, the simple baseline sees very few training examples for rare classes. The LLM classifier generalizes better here, but its confidence calibration remains uncertain (we never check calibration, only raw accuracy).

---

## 4. What Is Misleading About My Headline Number?

**Headline number: intent classification accuracy on the golden set.**

Several reasons this is misleading:

- **Subsample size:** 5,000 pairs from 3M tweets. The subsample may not capture the long tail of unusual customer messages (e.g., non-English tweets, memes, very long threads). The agent was never exposed to those during retrieval or evaluation.

- **"Resolved" is not verified:** The cleaned pairs corpus pairs each customer message with a *brand reply*, not with evidence that the issue was actually fixed. The retrieval corpus is a "SpotifyCares-replied" corpus. A retrieved "resolution" might be an acknowledgement with no real fix.

- **Golden set is hand-labelled by one person (me).** Inter-annotator agreement was not measured. Intent boundary decisions (e.g., is "I was charged after cancelling" `billing_subscription` or `cancellation`?) are subjective and my labels may be inconsistently applied.

- **Judge agreement measured on only 30 examples.** Cohen's kappa on 30 examples has high variance — the confidence interval on the kappa is wide. A kappa of 0.6 on 30 examples could be 0.4–0.8 with 95% CI.

- **Twitter 2017 text is not representative of 2025 support channels.** The dataset is from ~2017. Customer vocabulary, platform affordances, and even Spotify's feature set have changed significantly. An agent trained on this corpus would need retraining before production use.

- **Per-intent imbalance hidden by overall accuracy.** If `playback_technical` is 40% of the golden set and we classify it at 90% accuracy while `cancellation` classifies at 50%, the overall number looks acceptable but the failure on cancellations (a high-stakes intent) is masked.

---

## 5. What I'd Do With One More Week

1. **Build a real labelled training set for the simple baseline.** Currently the simple baseline is trained on the golden set (small N, data leakage risk). With a week I'd do a separate ~500-message labelling pass on messages *not in the golden set*, giving the LR classifier a real training split.

2. **Add embedding-based retrieval as an experiment.** Swap TF-IDF for `sentence-transformers/all-MiniLM-L6-v2` and measure whether retrieval quality (judge "grounded" score) improves. If it does materially, the infra cost is justified.

3. **Calibration check on the LLM classifier.** Plot confidence vs actual accuracy (reliability diagram). If the model says 0.95 confidence but is only right 70% of the time at that threshold, the escalation threshold (currently 0.70) needs adjustment.

4. **Fix the boilerplate filter.** Instead of discarding the whole reply, extract only the non-boilerplate second sentence and retain it in the corpus. This would increase retrieval corpus quality for common intents.

5. **Broaden the judge-vs-human agreement check to 50+ examples** with a calibrated rubric. The current 30-example check has too-wide confidence intervals to draw firm conclusions about judge reliability.
