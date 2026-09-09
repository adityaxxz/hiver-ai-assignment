"""
retrieval.py — Phase 2

Builds a TF-IDF index over historical customer_message texts and provides
cosine-similarity lookup to retrieve the top-k most similar past cases.

Design decision: TF-IDF over embeddings — zero extra infra, instant to
explain live, and good enough for grounding a draft reply. See decision log.

Usage (smoke check):
    python src/retrieval.py
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PARQUET_PATH = Path("data/cleaned_pairs.parquet")


class CaseRetriever:
    """
    Wraps a TF-IDF vectorizer fit on historical customer messages.
    Call .retrieve(query, k) to get the top-k similar (customer_msg, brand_reply) pairs.
    """

    def __init__(self, pairs: pd.DataFrame):
        self._pairs = pairs.reset_index(drop=True)
        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(self._pairs["customer_message"])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return top-k similar past cases as list of {customer_message, brand_reply, score}."""
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            results.append(
                {
                    "customer_message": self._pairs.loc[idx, "customer_message"],
                    "brand_reply": self._pairs.loc[idx, "brand_reply"],
                    "score": float(sims[idx]),
                }
            )
        return results


def load_retriever(parquet_path: Path = PARQUET_PATH) -> CaseRetriever:
    """Load cleaned pairs and build a CaseRetriever. Cheap: takes <1s on 5k pairs."""
    pairs = pd.read_parquet(parquet_path)
    return CaseRetriever(pairs)


if __name__ == "__main__":
    # Smoke check: retrieve 3 cases for a sample query and eyeball them
    retriever = load_retriever()
    test_queries = [
        "my spotify keeps crashing every time I try to play a song",
        "I was charged twice this month, I need a refund",
        "how do I download songs for offline listening?",
        "I can't log into my account, it says my password is wrong",
        "please cancel my premium subscription",
    ]
    for q in test_queries:
        print(f"\nQuery: {q!r}")
        for i, case in enumerate(retriever.retrieve(q, k=3), 1):
            print(f"  [{i}] score={case['score']:.3f}")
            print(f"       past_msg: {case['customer_message'][:80]!r}")
            print(f"       past_rep: {case['brand_reply'][:80]!r}")
