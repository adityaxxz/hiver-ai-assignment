# Decision Log

10–15 non-obvious decisions made during the build. Each entry states the decision and the reasoning.

---

1. **Chose SpotifyCares over AmazonHelp.**
   SpotifyCares has a small, coherent intent space (account/billing/playback/cancellation/features/praise) that a non-domain-expert can hand-label accurately in 150–250 examples. AmazonHelp mixes product lines (electronics, groceries, logistics) making a clean intent taxonomy nearly impossible. No financial-account PII risk with Spotify keeps the take-home clean.

2. **Reconstructed threads via `in_response_to_tweet_id` / `response_tweet_id` chains rather than trusting a single column.**
   The raw CSV has `response_tweet_id` (multi-valued, comma-separated in some rows) *and* `in_response_to_tweet_id`. Either column alone misses edges. Strategy: build a tweet_id index, then for each SpotifyCares reply walk `in_response_to_tweet_id` to find the parent customer message. Also pick up rows where `text` mentions `@SpotifyCares` directly, to catch thread roots.

3. **Subsampled to 5,000 pairs (fixed seed=42) rather than processing all 3M rows.**
   Full dataset would take ~30 min to load on a laptop and saturate RAM. 5k pairs gives hundreds of retrieval candidates per intent class — more than enough for TF-IDF grounding. Exact method: `df.sample(n=5000, random_state=42)` after deduplication. Decision log entry as required by assignment.

4. **Hand-built the intent taxonomy after reading real messages; used clustering only as a scouting aid.**
   Ran TF-IDF + KMeans(k=10) on a 500-message sample to see rough topic groupings, then discarded the cluster labels and wrote the 8-class taxonomy by hand. Clustering reliably surfaced "login/account," "crash/buffer," "charge/refund," and "cancel" as distinct groups — confirmed the taxonomy was not invented from thin air. Pruned rare edge cases into `other` and `complaint_general`.

5. **TF-IDF retrieval instead of embeddings.**
   Sentence-transformers require downloading a 400MB model; a vector DB (FAISS/Chroma) adds infra. TF-IDF cosine similarity over 5k pairs is fast (<50ms), explainable ("it matched on the words 'crash iPhone'"), and sufficient for grounding a draft. Would upgrade to embeddings if the retrieval failure analysis showed systematic topic mismatches at eval time.

6. **Three separate LLM calls per message (classify / draft / escalate), not one mega-prompt.**
   Keeps each prompt short enough to audit and explain live. Critically, it lets us evaluate each stage independently: a wrong classification and a bad draft are different failure modes that need separate metrics. One mega-prompt produces an entangled blob where you cannot isolate which sub-task failed.

7. **Escalation is a hybrid rule+LLM policy, not a pure LLM decision.**
   Deterministic signals are cheap and auditable: `billing_subscription` and `account_access` always escalate (money and account security require humans); low classifier confidence (<0.70) escalates; strong negative sentiment keywords escalate. These signals are *fed into* the LLM as explicit context, and the LLM writes the reason sentence. This mirrors Hiver's own product philosophy: AI handles grunt work, humans stay in control of anything risky. A pure-LLM escalation call is opaque and impossible to tune without rewriting the prompt.

8. **Used Groq free tier (OpenAI-compatible endpoint) with `gpt-oss-20b` for classify/escalate and `gpt-oss-120b` for draft/judge.**
   Cost/quality tradeoff: The 20b model is fast and sufficient for structured JSON classification/escalation. The 120b model produces better reasoning and prose for drafts and judge scores. Both stay within free-tier rate limits with retry backoff logic.

9. **"Historically resolved" only means "the brand replied," not "the issue was actually fixed."**
   The dataset has no ground-truth resolution signal (no CSAT score, no "issue closed" flag). The cleaned pairs corpus is more precisely a "brand-replied" corpus. This limitation is explicit in the report's "what's misleading" section — not hidden in a footnote.

10. **Golden set stratified by intent, not pure random.**
    Pure random would over-represent `playback_technical` and `complaint_general` (the highest-volume intents) and potentially produce zero `cancellation` or `praise` examples. Sampling ~25–35 per intent class gives each class enough examples for meaningful per-class metrics.

11. **Reported per-intent metrics, not just an overall accuracy number.**
    An 80% overall accuracy can hide that `cancellation` classifies at 40% and `other` classifies at 20%. Per-intent breakdown is mandatory for the report's failure analysis and for the "what's misleading" section.

12. **Measured judge-vs-human agreement on a 30-example subset instead of trusting the judge blindly.**
    An LLM judge can be systematically biased (e.g., prefers longer replies, or gives high tone scores to any reply mentioning "Hi"). Computing Cohen's kappa and within-1 agreement against a human-scored sample validates whether the judge's scores mean anything. Assignment explicitly requires this as a named deliverable.

13. **No pytest/CI test suite.**
    Optimized for "can explain and modify this live." Each module has a lightweight `if __name__ == "__main__"` smoke check. A full test suite would add 200+ lines of boilerplate that obscures the actual logic and would need to be explained during the live review. Assignment explicitly says to keep it legible.

14. **False auto-handle costs more than unnecessary escalate.**
    A false auto-handle on a `billing_subscription` or `account_access` issue means a customer with a real money or security problem is given a bot reply and left without a human. An unnecessary escalate wastes a human agent's time but causes no real harm. Escalation policy is tuned to minimize false auto-handles: these intent classes are always-escalate by rule, not by LLM judgment.

