# Eval Results Summary

## Intent Classification

| System | Accuracy | Macro-F1 |
|--------|----------|----------|
| trivial_baseline | 0.142 | 0.031 |
| simple_baseline | 0.375 | 0.306 |
| llm_pipeline | 0.517 | 0.467 |

## Escalation Decision

| System | Accuracy | Precision | Recall | False Auto-Handle | Unnecessary Escalate |
|--------|----------|-----------|--------|-------------------|----------------------|
| trivial_baseline | 0.408 | 0.408 | 1.000 | 0 | 71 |
| simple_baseline | 0.408 | 0.408 | 1.000 | 0 | 71 |
| llm_pipeline | 0.800 | 0.791 | 0.694 | 15 | 9 |

## Reply Quality (LLM Judge, avg 1-5)

| System | Grounded | Relevant | Tone |
|--------|----------|----------|------|
| trivial_baseline | N/A | N/A | N/A |
| simple_baseline | 2.7333333333333334 | 2.816666666666667 | 2.6333333333333333 |
| llm_pipeline | 3.775 | 3.808333333333333 | 3.533333333333333 |

## Judge-vs-Human Agreement (30-example subset)

- **grounded**: kappa=-0.273, within-1=100.0% (n=14)
- **relevant**: kappa=-0.185, within-1=78.6% (n=14)
- **tone**: kappa=0.013, within-1=64.3% (n=14)