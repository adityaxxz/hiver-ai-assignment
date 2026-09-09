"""
eval_harness.py — Phase 5/6

Runs all three systems (trivial baseline, simple baseline, full LLM pipeline)
against the golden set (golden_set.csv) and produces:
  - Intent classification: accuracy + macro-F1 + confusion matrix (per-intent)
  - Escalation: accuracy, precision, recall, cost analysis (false auto_handle vs unnecessary escalate)
  - Reply quality: LLM-judge scores (grounded/relevant/tone 1-5) for all examples
  - Judge-vs-human agreement: Cohen's kappa on the 30-example human-scored subset

Output:
  eval/results.json          — machine-readable results
  eval/results_summary.md    — human-readable summary table

Usage:
    python src/eval_harness.py [--skip-llm] [--judge-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Import from src/ (run from project root)
sys.path.insert(0, str(Path(__file__).parent))
from agent import (
    INTENTS,
    MODEL_SMART,
    _call_llm,
    _parse_json_with_retry,
    classify_intent,
    decide_escalation,
)
from baselines import SimpleBaseline, TrivialBaseline
from pipeline import run_agent
from retrieval import load_retriever

GOLDEN_SET = Path("golden_set.csv")
EVAL_DIR = Path("eval")
RESULTS_JSON = EVAL_DIR / "results.json"
RESULTS_MD = EVAL_DIR / "results_summary.md"

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are grading a customer-support draft reply. Score 1-5 on each: "
    "grounded (did it use the retrieved history sensibly, not invent things), "
    "relevant (addresses the actual message), "
    "tone (matches a real brand support voice, concise, not robotic). "
    'Respond with only JSON: {"grounded": <1-5>, "relevant": <1-5>, "tone": <1-5>, "rationale": "<one sentence>"}.'
)


def llm_judge(message: str, draft: str, similar_cases: list[dict]) -> dict:
    """Score a draft reply 1-5 on grounded/relevant/tone."""
    context_summary = "; ".join(
        f'"{c["customer_message"][:80]}" -> "{c["brand_reply"][:80]}"'
        for c in similar_cases[:2]
    )
    user = (
        f'Customer message: "{message}"\n'
        f'Draft reply: "{draft}"\n'
        f'Retrieved historical context used: "{context_summary}"'
    )
    result = _parse_json_with_retry(MODEL_SMART, JUDGE_SYSTEM, user, temperature=0)
    for key in ("grounded", "relevant", "tone"):
        result[key] = int(result.get(key, 3))
    return result


# ---------------------------------------------------------------------------
# Run a system over the golden set
# ---------------------------------------------------------------------------
def run_system_on_golden(
    golden: pd.DataFrame,
    system_name: str,
    predict_fn,
    use_llm_judge: bool = True,
    rate_limit_sleep: float = 1.0,
) -> list[dict]:
    """
    Run predict_fn(message) for each golden row.
    Returns list of result dicts with ground-truth appended.
    """
    rows = []
    for _, row in tqdm(golden.iterrows(), total=len(golden), desc=system_name):
        msg = row["customer_message"]
        try:
            pred = predict_fn(msg)
        except Exception as e:
            pred = {
                "intent": "other",
                "confidence": 0.0,
                "reply": "",
                "decision": {"action": "escalate", "reason": str(e), "signals": {}},
                "similar_cases": [],
            }

        judge_scores = {}
        if use_llm_judge and pred.get("reply"):
            try:
                judge_scores = llm_judge(msg, pred["reply"], pred.get("similar_cases", []))
                time.sleep(rate_limit_sleep)
            except Exception as e:
                judge_scores = {"grounded": -1, "relevant": -1, "tone": -1, "rationale": str(e)}

        rows.append(
            {
                "system": system_name,
                "message": msg,
                "true_intent": row["true_intent"],
                "pred_intent": pred["intent"],
                "true_escalate": row["should_escalate"],
                "pred_escalate": pred["decision"]["action"],
                "reply": pred.get("reply", ""),
                **judge_scores,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    intents_present = df["true_intent"].unique().tolist()

    # Intent classification
    intent_acc = accuracy_score(df["true_intent"], df["pred_intent"])
    intent_f1 = f1_score(df["true_intent"], df["pred_intent"], average="macro", zero_division=0)
    per_intent_f1 = f1_score(
        df["true_intent"], df["pred_intent"], labels=intents_present, average=None, zero_division=0
    )
    per_intent = {label: float(f) for label, f in zip(intents_present, per_intent_f1)}

    # Escalation
    # Map to binary: "escalate" = 1, "auto_handle" = 0
    true_esc = (df["true_escalate"] == "escalate").astype(int)
    pred_esc = (df["pred_escalate"] == "escalate").astype(int)
    esc_acc = accuracy_score(true_esc, pred_esc)
    esc_prec = precision_score(true_esc, pred_esc, zero_division=0)
    esc_rec = recall_score(true_esc, pred_esc, zero_division=0)
    # False auto_handle = true=escalate but pred=auto_handle (dangerous)
    false_auto_handle = int(((true_esc == 1) & (pred_esc == 0)).sum())
    unnecessary_escalate = int(((true_esc == 0) & (pred_esc == 1)).sum())

    # Reply quality (only rows where judge ran, score != -1)
    judge_df = df[df.get("grounded", pd.Series(dtype=float)).fillna(-1) > 0] if "grounded" in df else pd.DataFrame()
    avg_judge = {}
    if not judge_df.empty:
        for key in ("grounded", "relevant", "tone"):
            avg_judge[key] = float(judge_df[key].mean())

    return {
        "intent_accuracy": float(intent_acc),
        "intent_macro_f1": float(intent_f1),
        "per_intent_f1": per_intent,
        "escalation_accuracy": float(esc_acc),
        "escalation_precision": float(esc_prec),
        "escalation_recall": float(esc_rec),
        "false_auto_handle_count": false_auto_handle,
        "unnecessary_escalate_count": unnecessary_escalate,
        "avg_judge_scores": avg_judge,
    }


# ---------------------------------------------------------------------------
# Judge-vs-human agreement (30-example subset)
# ---------------------------------------------------------------------------
def judge_vs_human_agreement(golden: pd.DataFrame, llm_rows: list[dict]) -> dict:
    """
    golden must have columns: human_grounded, human_relevant, human_tone (optional 30-row subset).
    Computes Cohen's kappa and within-1 % agreement for each dimension.
    """
    # Only rows where human scores are provided
    human_cols = ["human_grounded", "human_relevant", "human_tone"]
    if not all(c in golden.columns for c in human_cols):
        return {"note": "No human scores in golden_set.csv; skipping judge-vs-human check."}

    subset = golden.dropna(subset=human_cols).head(30)
    if subset.empty:
        return {"note": "No human-scored rows found."}

    row_map = {r["message"]: r for r in llm_rows}
    results = {}
    for dim, hcol in zip(("grounded", "relevant", "tone"), human_cols):
        human_scores, judge_scores = [], []
        for _, grow in subset.iterrows():
            msg = grow["customer_message"]
            if msg not in row_map or row_map[msg].get(dim, -1) <= 0:
                continue
            human_scores.append(int(grow[hcol]))
            judge_scores.append(int(row_map[msg][dim]))
        if len(human_scores) < 2:
            results[dim] = {"note": "too few samples"}
            continue
        kappa = float(cohen_kappa_score(human_scores, judge_scores))
        within1 = float(
            sum(abs(h - j) <= 1 for h, j in zip(human_scores, judge_scores)) / len(human_scores)
        )
        results[dim] = {"kappa": kappa, "within_1_pct": within1, "n": len(human_scores)}
    return results


# ---------------------------------------------------------------------------
# Markdown summary writer
# ---------------------------------------------------------------------------
def write_markdown_summary(all_metrics: dict, agreement: dict, out_path: Path):
    lines = ["# Eval Results Summary\n"]
    lines.append("## Intent Classification\n")
    lines.append("| System | Accuracy | Macro-F1 |")
    lines.append("|--------|----------|----------|")
    for sys_name, m in all_metrics.items():
        lines.append(f"| {sys_name} | {m['intent_accuracy']:.3f} | {m['intent_macro_f1']:.3f} |")

    lines.append("\n## Escalation Decision\n")
    lines.append("| System | Accuracy | Precision | Recall | False Auto-Handle | Unnecessary Escalate |")
    lines.append("|--------|----------|-----------|--------|-------------------|----------------------|")
    for sys_name, m in all_metrics.items():
        lines.append(
            f"| {sys_name} | {m['escalation_accuracy']:.3f} | {m['escalation_precision']:.3f} | "
            f"{m['escalation_recall']:.3f} | {m['false_auto_handle_count']} | {m['unnecessary_escalate_count']} |"
        )

    lines.append("\n## Reply Quality (LLM Judge, avg 1-5)\n")
    lines.append("| System | Grounded | Relevant | Tone |")
    lines.append("|--------|----------|----------|------|")
    for sys_name, m in all_metrics.items():
        j = m.get("avg_judge_scores", {})
        lines.append(
            f"| {sys_name} | {j.get('grounded', 'N/A')} | {j.get('relevant', 'N/A')} | {j.get('tone', 'N/A')} |"
        )

    lines.append("\n## Judge-vs-Human Agreement (30-example subset)\n")
    for dim, res in agreement.items():
        if isinstance(res, dict) and "kappa" in res:
            lines.append(f"- **{dim}**: kappa={res['kappa']:.3f}, within-1={res['within_1_pct']:.1%} (n={res['n']})")
        else:
            lines.append(f"- **{dim}**: {res}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] Written: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(skip_llm: bool = False, judge_only: bool = False):
    if not GOLDEN_SET.exists():
        sys.exit(f"[eval] {GOLDEN_SET} not found. Build the golden set first.")

    golden = pd.read_csv(GOLDEN_SET)
    required_cols = {"customer_message", "true_intent", "should_escalate"}
    missing = required_cols - set(golden.columns)
    if missing:
        sys.exit(f"[eval] golden_set.csv missing columns: {missing}")

    EVAL_DIR.mkdir(exist_ok=True)

    # Load training slice for simple baseline (all golden rows NOT in the eval subset,
    # or if golden is small, use the full cleaned_pairs for retrieval-only training)
    pairs = pd.read_parquet(Path("data/cleaned_pairs.parquet"))

    # Train simple baseline on non-golden rows from pairs
    # (pairs has no labels, so we use golden labels for LR training — note in report: small N)
    train_golden = golden.sample(frac=0.5, random_state=42)
    eval_golden = golden.drop(train_golden.index)

    trivial = TrivialBaseline().fit(golden["true_intent"].tolist())

    simple = SimpleBaseline()
    simple.fit(train_golden["customer_message"].tolist(), train_golden["true_intent"].tolist())
    simple.load_retriever()

    all_results: dict[str, list[dict]] = {}
    all_metrics: dict[str, dict] = {}

    systems: list[tuple[str, callable, bool]] = [
        ("trivial_baseline", trivial.predict, False),
        ("simple_baseline", simple.predict, True),
    ]
    if not skip_llm:
        if "GROQ_API_KEY" not in os.environ:
            print("[eval] GROQ_API_KEY not set; skipping LLM pipeline.")
        else:
            systems.append(("llm_pipeline", run_agent, True))

    for sys_name, predict_fn, use_judge in systems:
        print(f"\n[eval] Running {sys_name} on {len(eval_golden)} golden examples …")
        rows = run_system_on_golden(
            eval_golden, sys_name, predict_fn,
            use_llm_judge=(use_judge and not skip_llm and "GROQ_API_KEY" in os.environ),
        )
        all_results[sys_name] = rows
        all_metrics[sys_name] = compute_metrics(rows)

    # Judge-vs-human agreement (uses LLM pipeline results if available)
    llm_rows = all_results.get("llm_pipeline", [])
    agreement = judge_vs_human_agreement(golden, llm_rows)

    RESULTS_JSON.write_text(
        json.dumps({"metrics": all_metrics, "agreement": agreement}, indent=2),
        encoding="utf-8",
    )
    print(f"[eval] Saved {RESULTS_JSON}")
    write_markdown_summary(all_metrics, agreement, RESULTS_MD)

    # Print quick summary
    for sys_name, m in all_metrics.items():
        print(
            f"\n{sys_name}: intent_acc={m['intent_accuracy']:.3f} "
            f"f1={m['intent_macro_f1']:.3f} "
            f"esc_acc={m['escalation_accuracy']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM pipeline and judge calls")
    parser.add_argument("--judge-only", action="store_true", help="Re-run judge on saved results")
    args = parser.parse_args()
    main(skip_llm=args.skip_llm, judge_only=args.judge_only)
