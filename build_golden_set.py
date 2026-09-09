"""
build_golden_set.py — one-time script to generate golden set candidates

Strategy (per PROJECT_CONTEXT.md §8):
  1. Keyword-heuristic pre-tag every cleaned_pair into one of 8 intents
  2. Stratified sample ~30 per intent (240 total)
  3. Save golden_set_candidates.csv for human review + correction
  4. Human corrects labels, fills should_escalate, reply_checklist
  5. Rename to golden_set.csv when done

Run from project root:
    python build_golden_set.py
"""

import re
import pandas as pd

PARQUET = "data/cleaned_pairs.parquet"
OUT = "golden_set_candidates.csv"

# ---------------------------------------------------------------------------
# Keyword heuristics per intent (checked in priority order)
# These are scouts, not classifiers — expect ~70% accuracy before hand-correction
# ---------------------------------------------------------------------------
RULES = [
    ("account_access", [
        r"\b(log\s?in|login|log\s?out|logout|sign\s?in|sign\s?out|password|reset.pass|account.lock|locked.out|can'?t.access|forgot.pass|username|email.not.found|account.not.found|verify|verification|two.factor|2fa)\b"
    ]),
    ("billing_subscription", [
        r"\b(charg|billed|bill|invoice|refund|payment|paid|price|cost|fee|premium.free|free.trial|receipt|subscription.fee|charged.twice|double.charged|money|credit.card|debit|bank|subscription.*cancel|cancel.*subscription|upgrade|downgrade|plan)\b"
    ]),
    ("cancellation", [
        r"\b(cancel|cancell|unsubscrib|stop.*subscription|end.*subscription|quit.*premium|leave.*premium|how.*cancel|want.*cancel|please.*cancel|cancel.*account|delete.*account)\b"
    ]),
    ("playback_technical", [
        r"\b(crash|crashing|buffer|buffering|lag|freeze|froz|not.play|won'?t.play|can'?t.play|skip|skipping|song.not|music.not|load|loading|offline|download.*song|song.*download|app.*not.work|not.work.*app|error|glitch|bug|broken|502|503|500|playback|shuffle|repeat|queue|playlist.*not|album.*not|slow|stuck)\b"
    ]),
    ("feature_question", [
        r"\b(how.do.i|how.can.i|how.to|where.is|where.can|what.is|what.are|does.spotify|can.spotify|is.there.a|how.*work|feature|find|search|discover|crossfade|equalizer|eq|podcast|lyrics|share|collaborate|family.plan|student|discount|available.on|support.*device|work.*on)\b"
    ]),
    ("praise", [
        r"\b(love|amazing|great|awesome|excellent|fantastic|best|thank.you|thanks|appreciate|brilliant|perfect|wonderful|impressed|happy.with|enjoy|enjoying|glad|favourite|favorite|good.job|well.done|keep.it.up)\b"
    ]),
    ("complaint_general", [
        r"\b(terrible|horrible|awful|worst|hate|useless|garbage|trash|ridiculous|disappoint|frustrat|annoying|unacceptable|pathetic|joke|disgust|fed.up|sick.of|tired.of|never.again|worst.*experience|bad.*experience|poor)\b"
    ]),
]

ESCALATION_RULES = {
    "billing_subscription": "escalate",
    "account_access": "escalate",
    "cancellation": "escalate",
    "playback_technical": "auto_handle",
    "feature_question": "auto_handle",
    "complaint_general": "auto_handle",
    "praise": "auto_handle",
    "other": "auto_handle",
}

REPLY_CHECKLISTS = {
    "account_access":       "Acknowledge login issue; suggest password reset or account recovery; escalate if unresolved",
    "billing_subscription": "Acknowledge charge issue; do NOT promise refund; direct to billing team; give timeline expectation",
    "cancellation":         "Explain cancellation steps; confirm what happens post-cancel (reverts to free); mention premium features lost",
    "playback_technical":   "Acknowledge problem; suggest quick fix (reinstall / clear cache / check connection); ask device/OS if needed",
    "feature_question":     "Answer the question directly; point to help article or in-app feature; keep it brief",
    "complaint_general":    "Acknowledge frustration genuinely; thank for feedback; no false promises; optionally offer to investigate",
    "praise":               "Acknowledge warmly; keep it short and genuine; no upsell",
    "other":                "Answer the specific question; point to help center if unsure",
}


def keyword_tag(text: str) -> str:
    t = text.lower()
    for intent, patterns in RULES:
        for pat in patterns:
            if re.search(pat, t):
                return intent
    return "other"


def main():
    df = pd.read_parquet(PARQUET)
    df = df[df["customer_message"].str.len() > 15].copy()  # drop very short messages
    df["auto_intent"] = df["customer_message"].apply(keyword_tag)

    print("Auto-tag distribution:")
    print(df["auto_intent"].value_counts().to_string())

    TARGET_PER_CLASS = 30
    samples = []
    for intent in ["account_access", "billing_subscription", "playback_technical",
                   "cancellation", "feature_question", "complaint_general", "praise", "other"]:
        bucket = df[df["auto_intent"] == intent]
        n = min(TARGET_PER_CLASS, len(bucket))
        if n == 0:
            print(f"WARNING: no samples for intent '{intent}'")
            continue
        sampled = bucket.sample(n=n, random_state=42)
        samples.append(sampled)
        print(f"  {intent}: sampled {n} (pool={len(bucket)})")

    golden = pd.concat(samples, ignore_index=True)

    # Add columns for human review
    golden["true_intent"] = golden["auto_intent"]           # HUMAN: correct this
    golden["should_escalate"] = golden["true_intent"].map(ESCALATION_RULES)  # HUMAN: verify
    golden["reply_checklist"] = golden["true_intent"].map(REPLY_CHECKLISTS)  # HUMAN: adjust
    golden["human_grounded"] = ""   # fill for 30-example judge agreement check
    golden["human_relevant"] = ""
    golden["human_tone"] = ""

    out_cols = [
        "customer_message", "brand_reply", "true_intent", "should_escalate",
        "reply_checklist", "human_grounded", "human_relevant", "human_tone",
        "thread_id"
    ]
    golden[out_cols].to_csv(OUT, index=False)
    print(f"\nSaved {len(golden)} candidates -> {OUT}")
    print("\nNEXT STEPS (do this by hand):")
    print("  1. Open golden_set_candidates.csv in Excel / VS Code")
    print("  2. Read each customer_message carefully")
    print("  3. Correct 'true_intent' where the keyword tag is wrong")
    print("  4. Correct 'should_escalate' (escalate/auto_handle) using your judgment")
    print("  5. For the first 30 rows: fill human_grounded/relevant/tone (1-5) AFTER running eval")
    print("  6. Rename file to golden_set.csv when done")


if __name__ == "__main__":
    main()
