"""
data_prep.py — Phase 1

Reads the raw Kaggle CSV (data/raw/twcs.csv), filters to SpotifyCares threads,
reconstructs (customer_message, brand_reply) pairs, cleans text, subsamples,
and saves data/cleaned_pairs.parquet.

Usage:
    python src/data_prep.py                    # uses defaults
    python src/data_prep.py --max-pairs 5000   # cap subsample size
"""

import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root, where this is invoked from)
# ---------------------------------------------------------------------------
RAW_CSV = Path("data/raw/twcs.csv")
OUT_PARQUET = Path("data/cleaned_pairs.parquet")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRAND = "SpotifyCares"
RANDOM_SEED = 42
# Subsample cap: we do NOT process the full 3M rows; state this in decision log
DEFAULT_MAX_PAIRS = 5000

# Boilerplate replies from SpotifyCares that add zero signal — skip them.
# Identified by reading ~50 brand replies manually.
BOILERPLATE_PATTERNS = [
    r"please (dm|message|contact) us",
    r"we('re| are) sorry to hear",
    r"thanks? for reaching out",
    r"^hi(,|!| there)",
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Strip routing @mentions, URLs, unescape HTML entities, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove @mentions used only for routing (leading @handle at start or after whitespace)
    text = re.sub(r"@\w+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_boilerplate(text: str) -> bool:
    """Return True if the brand reply is templated/boilerplate with no real content."""
    return bool(_BOILERPLATE_RE.search(text)) and len(text.split()) < 20


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    """Load CSV with minimal dtypes. Handles the Kaggle twcs.csv schema."""
    print(f"[data_prep] Loading {path} …", flush=True)
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": str,        # 'True'/'False' strings in raw CSV
            "created_at": str,
            "text": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        low_memory=False,
    )
    print(f"[data_prep] Loaded {len(df):,} rows", flush=True)
    return df


def filter_spotify_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that are part of a SpotifyCares thread:
    - All rows authored by SpotifyCares (brand replies)
    - All inbound (customer) rows that either mention @SpotifyCares or
      are a reply in a thread where SpotifyCares also appears.

    Strategy: collect all tweet_ids SpotifyCares replied to (in_response_to),
    then keep all rows whose tweet_id is reachable from those.
    """
    df = df.copy()
    df["inbound"] = df["inbound"].map({"True": True, "False": False, True: True, False: False})

    brand_mask = df["author_id"] == BRAND
    brand_df = df[brand_mask]
    print(f"[data_prep] SpotifyCares rows: {len(brand_df):,}", flush=True)

    # tweet_ids that SpotifyCares directly replied to
    direct_targets = set(brand_df["in_response_to_tweet_id"].dropna().unique())

    # Keep: brand rows + all customer rows that are in SpotifyCares' reply chain
    # Also keep rows where text mentions @SpotifyCares
    mentions_brand = df["text"].str.contains("@SpotifyCares", case=False, na=False)
    relevant_ids = set(df.loc[brand_mask, "tweet_id"]) | direct_targets

    relevant_mask = brand_mask | df["tweet_id"].isin(relevant_ids) | mentions_brand
    result = df[relevant_mask].copy()
    print(f"[data_prep] Spotify-thread rows: {len(result):,}", flush=True)
    return result


def reconstruct_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each brand reply row, find the customer message it replied to.
    Returns DataFrame with columns:
        thread_id, customer_message, brand_reply, timestamp
    """
    id_to_row = df.set_index("tweet_id")

    pairs = []
    brand_replies = df[df["author_id"] == BRAND].copy()

    for _, brand_row in brand_replies.iterrows():
        parent_id = brand_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id):
            continue
        if parent_id not in id_to_row.index:
            continue

        parent = id_to_row.loc[parent_id]
        # parent should be inbound (customer) — skip brand-to-brand rows
        if not parent.get("inbound", False):
            continue

        customer_msg = clean_text(str(parent.get("text", "")))
        brand_reply_text = clean_text(str(brand_row.get("text", "")))

        if not customer_msg or not brand_reply_text:
            continue
        if is_boilerplate(brand_reply_text):
            continue

        pairs.append(
            {
                "thread_id": brand_row["tweet_id"],
                "customer_message": customer_msg,
                "brand_reply": brand_reply_text,
                "timestamp": brand_row.get("created_at", ""),
            }
        )

    result = pd.DataFrame(pairs)
    print(f"[data_prep] Reconstructed pairs (pre-dedup): {len(result):,}", flush=True)
    return result


def deduplicate(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drop near-identical customer messages (same first 80 chars)."""
    pairs = pairs.copy()
    pairs["_key"] = pairs["customer_message"].str[:80]
    pairs = pairs.drop_duplicates(subset="_key").drop(columns="_key")
    print(f"[data_prep] After dedup: {len(pairs):,}", flush=True)
    return pairs


def subsample(pairs: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    """Deterministic subsample. If already under max_pairs, return as-is."""
    if len(pairs) <= max_pairs:
        return pairs
    sampled = pairs.sample(n=max_pairs, random_state=RANDOM_SEED)
    print(f"[data_prep] Subsampled to {len(sampled):,} pairs (seed={RANDOM_SEED})", flush=True)
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(max_pairs: int = DEFAULT_MAX_PAIRS) -> pd.DataFrame:
    if not RAW_CSV.exists():
        sys.exit(
            f"[data_prep] ERROR: {RAW_CSV} not found.\n"
            "Download twcs.csv from Kaggle (thoughtvector/customer-support-on-twitter)\n"
            "and place it at data/raw/twcs.csv"
        )

    df_raw = load_raw(RAW_CSV)
    df_spotify = filter_spotify_rows(df_raw)
    pairs = reconstruct_pairs(df_spotify)
    pairs = deduplicate(pairs)
    pairs = subsample(pairs, max_pairs)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(OUT_PARQUET, index=False)
    print(f"[data_prep] Saved {len(pairs):,} pairs → {OUT_PARQUET}", flush=True)
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SpotifyCares support pairs")
    parser.add_argument(
        "--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
        help=f"Max pairs to keep after subsampling (default: {DEFAULT_MAX_PAIRS})"
    )
    args = parser.parse_args()
    pairs = main(max_pairs=args.max_pairs)
    print(pairs.head(3).to_string())
