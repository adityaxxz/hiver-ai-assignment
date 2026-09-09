"""
baselines.py — Phase 4

Two baselines to compare against the full LLM pipeline:

1. Trivial baseline:
   - Intent: always predicts the majority class from the golden set.
   - Reply: fixed canned string.
   - Decision: always "escalate".

2. Simple baseline:
   - Intent: TF-IDF + LogisticRegression trained on a labelled slice.
   - Reply: verbatim nearest historical brand_reply (no generation).
   - Decision: deterministic rule-only (same signals as agent.py, no LLM call).

Both expose a predict(message) interface returning the same dict shape as pipeline.run_agent().
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agent import ALWAYS_ESCALATE_INTENTS, CONFIDENCE_THRESHOLD, NEGATIVE_KEYWORDS
from retrieval import load_retriever

CANNED_REPLY = "Thanks for reaching out! Please follow up with more details and we'll take a look."


# ---------------------------------------------------------------------------
# Trivial baseline
# ---------------------------------------------------------------------------
class TrivialBaseline:
    """Always predicts majority intent, fixed reply, always escalates."""

    def __init__(self, majority_intent: str = "playback_technical"):
        self.majority_intent = majority_intent

    def fit(self, labels: list[str]) -> "TrivialBaseline":
        """Determine majority class from a list of intent labels."""
        from collections import Counter
        self.majority_intent = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, message: str) -> dict:
        return {
            "message": message,
            "intent": self.majority_intent,
            "confidence": 1.0,
            "similar_cases": [],
            "reply": CANNED_REPLY,
            "decision": {
                "action": "escalate",
                "reason": "Trivial baseline always escalates.",
                "signals": {},
            },
        }


# ---------------------------------------------------------------------------
# Simple baseline
# ---------------------------------------------------------------------------
class SimpleBaseline:
    """TF-IDF + LogisticRegression for intent; nearest brand_reply verbatim; rule-only escalation."""

    def __init__(self):
        self._clf = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)),
                ("lr", LogisticRegression(max_iter=1000, C=1.0)),
            ]
        )
        self._retriever = None
        self._fitted = False

    def fit(self, messages: list[str], labels: list[str]) -> "SimpleBaseline":
        self._clf.fit(messages, labels)
        self._fitted = True
        return self

    def load_retriever(self, parquet_path: Path = Path("data/cleaned_pairs.parquet")) -> "SimpleBaseline":
        self._retriever = load_retriever(parquet_path)
        return self

    def _rule_escalation(self, message: str, intent: str, confidence: float) -> dict:
        always = intent in ALWAYS_ESCALATE_INTENTS
        low_conf = confidence < CONFIDENCE_THRESHOLD
        neg = any(kw in message.lower() for kw in NEGATIVE_KEYWORDS)
        action: Literal["escalate", "auto_handle"] = (
            "escalate" if (always or low_conf or neg) else "auto_handle"
        )
        reasons = []
        if always:
            reasons.append(f"intent '{intent}' always requires human")
        if low_conf:
            reasons.append(f"low classifier confidence ({confidence:.2f})")
        if neg:
            reasons.append("negative sentiment detected")
        reason = "; ".join(reasons) if reasons else "no escalation signals"
        return {"action": action, "reason": reason, "signals": {"always_escalate_intent": always, "low_confidence": low_conf, "negative_sentiment": neg}}

    def predict(self, message: str) -> dict:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")
        proba = self._clf.predict_proba([message])[0]
        classes = self._clf.classes_
        intent = classes[proba.argmax()]
        confidence = float(proba.max())

        similar_cases = []
        reply = CANNED_REPLY
        if self._retriever:
            similar_cases = self._retriever.retrieve(message, k=1)
            reply = similar_cases[0]["brand_reply"] if similar_cases else CANNED_REPLY

        decision = self._rule_escalation(message, intent, confidence)
        return {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "similar_cases": similar_cases,
            "reply": reply,
            "decision": decision,
        }


if __name__ == "__main__":
    # Smoke check with dummy data
    messages = [
        "my spotify keeps crashing",
        "I was charged twice",
        "how do I cancel premium",
        "app won't load any songs",
        "I can't log in to my account",
    ]
    labels = ["playback_technical", "billing_subscription", "cancellation", "playback_technical", "account_access"]

    tb = TrivialBaseline().fit(labels)
    print("Trivial:", tb.predict(messages[0]))

    sb = SimpleBaseline().fit(messages, labels)
    print("Simple:", sb.predict(messages[0]))
