# Experiment Progress & Scoreboard

## Leaderboard & Cross-Validation Summary

Target Metric: **18-Class Balanced Accuracy** (unweighted mean of per-class recalls).
Stress Metric: **Dirichlet Prior-Shift 5th-Percentile Balanced Accuracy (p05)**.

| Exp ID | Model / Pipeline Description | 5-Fold Balanced Acc (Mean ± Std) | Dirichlet p05 | Public LB | Worst-3 Sibling Classes (Recall) | OOF Parquet Path | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Benchmark* | *Mizuno0522 baseline (README)* | 0.3260 | — | — | — | — | Baseline ref |
| **EXP-001** | Multi-granular TF-IDF (word 1-2, char 3-5) + 165 stylometric features + Fast Linear Ensemble (Ridge 55% + SGD Huber 25% + SGD Log 20%) | **0.2849** (±0.0131) | **0.2582** | **0.269** (Rank #2229) | glm-5.3-flash (0.13), gpt-5.6-terra (0.16), claude-opus-4-8 (0.17) | `artifacts/oof/exp001_tfidf_linear.parquet` | Verified |
| **EXP-002** | Two-Stage LoRA (8B+14B) + 4-Gaussian Stack + 10% ATLAS + Curly/EmDash Invariants + Hungarian 9-18 | **0.3682** (±0.0136) | **0.3416** | *Scoring Pending* (v3 Ref: 57476183; v6 Dual T4 GPU Committed & Ready) | claude-fable-5 (0.13), glm-5.3-flash (0.18), gpt-5.6-sol (0.18) | `artifacts/oof/exp002_apex_atlas.parquet` | v3 Scoring / v6 GPU Verified & Ready |
| **EXP-003** | Apex ATLAS + Temp 0.92 + Two-Tier DP Injective Marginals (2-8) & Hungarian (9-18) | **0.3682** (±0.0136) | **0.3416** | *Scoring Pending* (v3 Ref: 57476194; v6 Dual T4 GPU Committed & Ready) | claude-fable-5 (0.13), glm-5.3-flash (0.18), gpt-5.6-sol (0.18) | `artifacts/oof/exp003_twotier_injective.parquet` | v3 Scoring / v6 GPU Verified & Ready |
