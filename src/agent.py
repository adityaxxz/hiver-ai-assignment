"""
agent.py — Phase 2

Three separate LLM calls per message (classification, draft reply, escalation decision).
Uses Groq's OpenAI-compatible endpoint. See PROJECT_CONTEXT.md §5-6 for design rationale.

Environment variable required: GROQ_API_KEY
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # ensure GROQ_API_KEY is available even when not invoked via run_pipeline.py

# ---------------------------------------------------------------------------
# LLM client (Groq, OpenAI-compatible)
# ---------------------------------------------------------------------------
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _client


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_FAST = "openai/gpt-oss-20b"   # classify / escalate
MODEL_SMART = "openai/gpt-oss-120b"  # draft / judge

# ---------------------------------------------------------------------------
# Intent taxonomy (from PROJECT_CONTEXT.md §4)
# ---------------------------------------------------------------------------
INTENTS = [
    "account_access",
    "billing_subscription",
    "playback_technical",
    "cancellation",
    "feature_question",
    "complaint_general",
    "praise",
    "other",
]

# Intents that always trigger human escalation (money + account security)
ALWAYS_ESCALATE_INTENTS = {"billing_subscription", "account_access"}

# Confidence threshold below which we flag low_confidence
CONFIDENCE_THRESHOLD = 0.70

# Simple negative sentiment keywords (deterministic signal)
NEGATIVE_KEYWORDS = [
    "furious", "horrible", "terrible", "disgusting", "worst", "awful",
    "unacceptable", "scam", "fraud", "useless", "garbage", "ridiculous",
    "hate", "lied", "lies", "cheated", "stealing", "robbery",
]


# ---------------------------------------------------------------------------
# Helper: call LLM with retry on JSON parse failure and rate-limit backoff
# ---------------------------------------------------------------------------
def _call_llm(model: str, system: str, user: str, temperature: float = 0) -> str:
    """Call LLM; retry once on 429. Returns raw content string."""
    client = _get_client()
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e) and attempt == 0:
                time.sleep(5)
                continue
            raise


def _parse_json_with_retry(model: str, system: str, user: str, temperature: float = 0) -> dict:
    """Call LLM, parse JSON; retry once if parse fails."""
    for attempt in range(2):
        raw = _call_llm(model, system, user, temperature)
        # Strip markdown code fences if model wrapped JSON in them
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 0:
                continue
            raise ValueError(f"LLM returned unparseable JSON after retry: {raw!r}")


# ---------------------------------------------------------------------------
# Stage 1: classify_intent
# ---------------------------------------------------------------------------
CLASSIFY_SYSTEM = (
    "You are a support-ticket classifier for Spotify's customer support team. "
    "Classify the customer message into exactly one of these intents: "
    + ", ".join(INTENTS)
    + '. Respond with only JSON: {"intent": "<one of the labels>", "confidence": <0-1 float>}.'
)


def classify_intent(message: str) -> dict:
    """
    Returns: {"intent": str, "confidence": float}
    """
    user = f'Customer message: "{message}"'
    result = _parse_json_with_retry(MODEL_FAST, CLASSIFY_SYSTEM, user, temperature=0)
    # Validate
    if result.get("intent") not in INTENTS:
        result["intent"] = "other"
    result["confidence"] = float(result.get("confidence", 0.5))
    return result


# ---------------------------------------------------------------------------
# Stage 2: draft_reply
# ---------------------------------------------------------------------------
DRAFT_SYSTEM = (
    "You are drafting a reply as SpotifyCares support, in Spotify's real historical voice: "
    "brief, friendly, direct, no corporate filler. Ground your reply in how similar past issues "
    "were actually resolved below. If the retrieved cases don't actually match the customer's "
    "issue, say so plainly in the draft rather than inventing a resolution."
)


def draft_reply(message: str, intent: str, similar_cases: list[dict]) -> str:
    """
    Returns: draft reply text (plain string, not JSON).
    similar_cases: list of {"customer_message": ..., "brand_reply": ...}
    """
    cases_block = "\n".join(
        f'{i}. "{c["customer_message"][:200]}" -> "{c["brand_reply"][:200]}"'
        for i, c in enumerate(similar_cases[:3], 1)
    )
    user = (
        f'Customer message: "{message}"\n'
        f"Classified intent: {intent}\n"
        f"Similar past cases (customer message -> how SpotifyCares actually replied):\n"
        f"{cases_block}\n\n"
        "Write ONE reply to the current customer message."
    )
    return _call_llm(MODEL_SMART, DRAFT_SYSTEM, user, temperature=0.3).strip()


# ---------------------------------------------------------------------------
# Stage 3: decide_escalation
# ---------------------------------------------------------------------------
ESCALATE_SYSTEM = (
    "You decide whether a support message can be auto-handled by AI or must escalate to a human "
    "agent. You are given deterministic signals; weigh them, don't ignore them. "
    'Respond with only JSON: {"action": "auto_handle"|"escalate", "reason": "<one sentence, specific to this message>"}.'
)


def _compute_deterministic_signals(message: str, intent: str, confidence: float) -> dict:
    always_escalate = intent in ALWAYS_ESCALATE_INTENTS
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    msg_lower = message.lower()
    negative_sentiment = any(kw in msg_lower for kw in NEGATIVE_KEYWORDS)
    return {
        "always_escalate_intent": always_escalate,
        "low_confidence": low_confidence,
        "negative_sentiment": negative_sentiment,
    }


def decide_escalation(message: str, intent: str, confidence: float) -> dict:
    """
    Returns: {"action": "auto_handle"|"escalate", "reason": str,
              "signals": {always_escalate_intent, low_confidence, negative_sentiment}}
    """
    signals = _compute_deterministic_signals(message, intent, confidence)
    user = (
        f'Customer message: "{message}"\n'
        f"Intent: {intent}  (classifier confidence: {confidence:.2f})\n"
        f"Deterministic signals: "
        f"always_escalate_intent={signals['always_escalate_intent']}, "
        f"low_confidence={signals['low_confidence']}, "
        f"negative_sentiment={signals['negative_sentiment']}"
    )
    result = _parse_json_with_retry(MODEL_FAST, ESCALATE_SYSTEM, user, temperature=0)
    result["signals"] = signals
    return result


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "GROQ_API_KEY" not in os.environ:
        sys.exit("Set GROQ_API_KEY before running this smoke check.")

    sample = "my spotify app keeps crashing on my iPhone whenever I try to play a playlist"
    print("=== classify_intent ===")
    cls = classify_intent(sample)
    print(cls)

    print("\n=== draft_reply ===")
    mock_cases = [
        {"customer_message": "spotify crashes on my phone", "brand_reply": "Hi! Try reinstalling the app and clearing the cache. Let us know if that helps!"},
    ]
    draft = draft_reply(sample, cls["intent"], mock_cases)
    print(draft)

    print("\n=== decide_escalation ===")
    esc = decide_escalation(sample, cls["intent"], cls["confidence"])
    print(esc)
