"""
pipeline.py — Phase 2

Glue layer: loads retriever once, then exposes run_agent(message) which calls
classify → retrieve → draft → decide in sequence and returns a single dict.

Usage:
    python src/pipeline.py "my spotify won't play any songs"
"""

import sys
from pathlib import Path

from retrieval import load_retriever, CaseRetriever
from agent import classify_intent, draft_reply, decide_escalation

_retriever: CaseRetriever | None = None


def _get_retriever() -> CaseRetriever:
    global _retriever
    if _retriever is None:
        parquet = Path("data/cleaned_pairs.parquet")
        if not parquet.exists():
            raise FileNotFoundError(
                "data/cleaned_pairs.parquet not found. Run: python src/data_prep.py"
            )
        _retriever = load_retriever(parquet)
    return _retriever


def run_agent(message: str, k: int = 3) -> dict:
    """
    Full pipeline for one customer message.

    Returns:
        {
          "message": str,
          "intent": str,
          "confidence": float,
          "similar_cases": [...],
          "reply": str,
          "decision": {"action": str, "reason": str, "signals": dict}
        }
    """
    retriever = _get_retriever()

    # Stage 1: classify
    cls = classify_intent(message)
    intent = cls["intent"]
    confidence = cls["confidence"]

    # Stage 2: retrieve
    similar_cases = retriever.retrieve(message, k=k)

    # Stage 3: draft
    reply = draft_reply(message, intent, similar_cases)

    # Stage 4: escalate
    decision = decide_escalation(message, intent, confidence)

    return {
        "message": message,
        "intent": intent,
        "confidence": confidence,
        "similar_cases": similar_cases,
        "reply": reply,
        "decision": decision,
    }


if __name__ == "__main__":
    import json, os

    if "GROQ_API_KEY" not in os.environ:
        sys.exit("Set GROQ_API_KEY before running pipeline.")

    msg = " ".join(sys.argv[1:]) or "my spotify keeps crashing on my iPhone"
    print(f"Running agent on: {msg!r}\n")
    result = run_agent(msg)

    print(f"Intent:     {result['intent']} (confidence={result['confidence']:.2f})")
    print(f"Decision:   {result['decision']['action']} — {result['decision']['reason']}")
    print(f"\nDraft reply:\n{result['reply']}")
    print(f"\nTop similar case: {result['similar_cases'][0]['customer_message'][:80]!r}")
