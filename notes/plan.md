# Forward Plan & Hypothesis Backlog

All experiments must be prioritized by $\frac{\text{Expected CV Gain}}{\text{Compute Cost}}$.

## Active & Prioritized Hypotheses

| ID | Status | Priority | Compute Tier | Hypothesis & Implementation Target |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | `[COMPLETED]` | P0 | Local CPU | **Multi-granular TF-IDF (Word 1-2, Char 3-5) + 165 Stylometric features + Calibrated Linear Classifiers (Ridge 55% + SGD Huber 25% + SGD Log 20%).** CV: 0.2849, Prior-Shift p05: 0.2582, Public LB: 0.269 (Rank #2229). |
| **EXP-002** | `[SUBMITTED]` | P0 | Kaggle / GPU | **Two-Stage LoRA (8B+14B) + 4-Gaussian Stack + 10% ATLAS + Curly/EmDash Invariants + Hungarian 9-18.** CV: 0.3682, Prior-Shift p05: 0.3416, v3 scoring (Ref: 57476183), v6 Dual T4 GPU committed & verified (`CUDA=True, GPUs=2`, `KernelWorkerStatus.COMPLETE`). |
| **EXP-003** | `[SUBMITTED]` | P0 | Kaggle / GPU | **Apex ATLAS + Temp 0.92 + Two-Tier DP Injective Marginals (2-8) & Hungarian (9-18).** CV: 0.3682, Prior-Shift p05: 0.3416, v3 scoring (Ref: 57476194), v6 Dual T4 GPU committed & verified (`CUDA=True, GPUs=2`, `KernelWorkerStatus.COMPLETE`). |
| **EXP-004** | `[NEXT]` | P0 | Local CPU + Colab | **Hierarchical Intra-Family Sibling Specialists.** Dedicated 5-way OpenAI, 5-way Anthropic, and 4-way Gemini sub-classifiers targeting fine-grained sister model disambiguation (current worst: `claude-fable-5` 0.128, `glm-5.3-flash` 0.180). |
| **EXP-005** | `[QUEUED]` | P0 | Local CPU | **Dirichlet Prior-Shift Multiplier Calibration & Sinkhorn Marginal Balancing.** Test-time optimization of $\alpha_k$ multipliers regularized against test distribution shifts to maximize worst-case Balanced Accuracy. |
| **EXP-006** | `[QUEUED]` | P2 | Colab Pro CLI GPU | **Multi-Tokenizer Fertility & Cross-Entropy Perplexity Forensics.** Extract token compression and log-rank curvature across Qwen3, Gemma-2, and GLM open tokenizers. |
| **EXP-007** | `[COMPLETED]` | P2 | Kaggle CLI / MCP | **Offline Submission Pipeline Push & Kaggle Kernel Verification.** End-to-end verification of `rajucode/name-that-llm-exp001` offline scoring runtime (132.8s, <15 min limit) and test score generation. |
