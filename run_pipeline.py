#!/usr/bin/env python
"""
run_pipeline.py — single CLI entrypoint

Commands:
    python run_pipeline.py prep               # run data_prep.py
    python run_pipeline.py run "message"      # run agent on one message
    python run_pipeline.py sample [N]         # run agent on N random messages from cleaned_pairs
    python run_pipeline.py eval [--skip-llm]  # run full eval harness
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env into os.environ

sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_prep(args):
    import data_prep
    data_prep.main(max_pairs=args.max_pairs)


def cmd_run(args):
    if "GROQ_API_KEY" not in os.environ:
        sys.exit("Set GROQ_API_KEY first.")
    from pipeline import run_agent
    message = " ".join(args.message)
    print(f"Message: {message!r}\n")
    result = run_agent(message)
    print(f"Intent:     {result['intent']} (confidence={result['confidence']:.2f})")
    print(f"Decision:   {result['decision']['action']} — {result['decision']['reason']}")
    print(f"\nDraft reply:\n{result['reply']}")
    top = result["similar_cases"][0] if result["similar_cases"] else {}
    if top:
        print(f"\nTop retrieval (score={top['score']:.3f}):")
        print(f"  past_msg: {top['customer_message'][:100]!r}")
        print(f"  past_rep: {top['brand_reply'][:100]!r}")


def cmd_sample(args):
    if "GROQ_API_KEY" not in os.environ:
        sys.exit("Set GROQ_API_KEY first.")
    import pandas as pd
    from pipeline import run_agent

    parquet = Path("data/cleaned_pairs.parquet")
    if not parquet.exists():
        sys.exit("Run: python run_pipeline.py prep first.")
    pairs = pd.read_parquet(parquet)
    sample = pairs.sample(n=min(args.n, len(pairs)), random_state=42)

    results = []
    for _, row in sample.iterrows():
        msg = row["customer_message"]
        try:
            result = run_agent(msg)
        except Exception as e:
            result = {"message": msg, "intent": "error", "confidence": 0.0, "reply": str(e),
                      "decision": {"action": "escalate", "reason": str(e)}}
        results.append(result)
        print(f"\n--- Message: {msg[:80]!r}")
        print(f"    Intent:   {result['intent']} ({result['confidence']:.2f})")
        print(f"    Decision: {result['decision']['action']} — {result['decision']['reason']}")
        print(f"    Reply:    {result['reply'][:120]}")

    out = Path("eval/sample_run.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} results → {out}")


def cmd_eval(args):
    from eval_harness import main as eval_main
    eval_main(skip_llm=args.skip_llm, judge_only=getattr(args, "judge_only", False))


def main():
    parser = argparse.ArgumentParser(description="SpotifyCares AI support agent pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # prep
    p_prep = sub.add_parser("prep", help="Run data preparation")
    p_prep.add_argument("--max-pairs", type=int, default=5000)
    p_prep.set_defaults(func=cmd_prep)

    # run
    p_run = sub.add_parser("run", help="Run agent on a single message")
    p_run.add_argument("message", nargs="+", help="Customer message (quote it)")
    p_run.set_defaults(func=cmd_run)

    # sample
    p_sample = sub.add_parser("sample", help="Run agent on N random messages")
    p_sample.add_argument("n", nargs="?", type=int, default=10)
    p_sample.set_defaults(func=cmd_sample)

    # eval
    p_eval = sub.add_parser("eval", help="Run full eval harness")
    p_eval.add_argument("--skip-llm", action="store_true")
    p_eval.add_argument("--judge-only", action="store_true", help="Recompute judge-vs-human kappa from saved results (instant)")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
