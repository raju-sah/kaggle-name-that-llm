# AGENTS.md — The Constitution & Relay Protocol

## 1. Ground Truth & Competition Invariants
- **Competition:** Kaggle — Name That LLM (Featured Code Competition, Metric: Balanced Accuracy over 18 classes).
- **Target Contract:** Predict exact authoring LLM from 18 frontier models on unseen prompts.
- **Metric Formulation:**
  $$\text{Balanced Accuracy} = \frac{1}{18} \sum_{k=1}^{18} \frac{TP_k}{P_k}$$
  *Rule:* Never starve any class. A class with 0 predictions gets 0 recall, costing $\frac{1}{18} \approx 0.0556$ penalty.
- **Leak-Free Validation Invariant:** Test passages are from prompts **disjoint** from `train.csv`.
  *Red Line:* NEVER use standard KFold or StratifiedKFold. Always use locked `StratifiedGroupKFold(n_splits=5)` on `prompt_id` (`data/processed/cv_folds.parquet`).
- **Submission Output Contract:**
  - File: `submission.parquet` (Parquet format, not CSV).
  - Columns: exactly `['id', 'model']`.
  - Content: No nulls, no duplicate `id`s, length matches `test.csv`, all predicted labels strictly in the 18 canonical strings.
  - Offline environment: Internet is disabled during scoring. Runtime $\le$ 540 min.
- **Strict Pre-Submission Gate (Zero Wasted Quota Rule):**
  - **Subagents are FORBIDDEN from calling `kaggle competitions submit` or `competition_submit_code`.** Only the Lead Orchestrator may submit after verifying all gates below.
  - **Gate 1 (Hardware Metadata):** Verify via `GetKernel` that `_enable_gpu == True` and `_machine_shape == 'NvidiaTeslaT4'` (`GPUs=2` confirmed in commit log).
  - **Gate 2 (Full Code-Path Smoke Execution on Commit):** Never skip GPU forward passes with `if len(test) >= 50:` without running a mini-batch smoke pass (`train` subsampled to 2 rows/class = 36 rows when `len(test) < 50`) through every single model (`lora8b_v1`, `lora14b_v1`, `_GAUSSIAN_SOURCE`, `ATLAS`, `Sinkhorn`, `Invariants`, `Hungarian`) during the commit run. Every line of code that runs on the hidden test set MUST execute during commit.
  - **Gate 3 (Mount Verification):** Every attached dataset/kernel/model path must be verified inside the commit run on Kaggle before submission.

## 2. The 18 Canonical Model Labels & Providers
| Provider (6) | Canonical Label (`model`) |
| :--- | :--- |
| **OpenAI (5)** | `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra` |
| **Anthropic (5)** | `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5` |
| **Google (4)** | `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash` |
| **ZAI (2)** | `glm-5.3`, `glm-5.3-flash` |
| **Qwen (1)** | `qwen3.8-max` |
| **Moonshot (1)**| `kimi-k3` |

## 3. 3-Tier Compute Strategy
1. **Tier 1 (Local CPU):** Control plane, data download, locked CV creation, 120+ stylometric & TF-IDF feature extraction, fast Linear/GBDT baselines, OOF ensembling & prior-shift post-processing.
2. **Tier 2 (Colab Pro CLI GPU):** DeBERTa-v3 / ModernBERT contrastive fine-tuning and LLM perplexity extraction; fold weights pushed to Kaggle Dataset (`rajucode/name-that-llm-models`).
3. **Tier 3 (Kaggle Offline Scoring):** Fast inference (<15 min) in Kaggle Notebook reading pre-trained weights and writing `submission.parquet`.

## 4. Standard CLI & MCP Commands
```bash
# Data download
kaggle competitions download -c name-that-llm -p data/raw && unzip -o data/raw/name-that-llm.zip -d data/raw

# Kaggle kernel push / status / output
kaggle kernels push -p notebooks/
kaggle kernels status rajucode/<NOTEBOOK>
kaggle kernels output rajucode/<NOTEBOOK> -p artifacts/kernel_output/

# Kaggle competition submit
kaggle competitions submit -c name-that-llm -f submission.parquet -k rajucode/<NOTEBOOK> -v <VERSION> -m "CV: 0.XXXX | Ensemble details"
```

## 5. 60-Second Wakeup Protocol (For Any Entering Agent)
1. Read `AGENTS.md` (this file) for rules and invariants.
2. Read `notes/plan.md` to identify the current top `[NEXT]` hypothesis.
3. Check `notes/closed_ideas.md` to avoid repeating graveyard ideas.
4. Review `notes/progress.md` for current CV benchmarks and OOF status.
5. Use existing recipes in `cookbooks/` instead of reinventing boilerplate.

## 6. Pass-The-Baton Checklist (Before Ending Any Turn)
- [ ] OOF predictions saved to `artifacts/oof/expXXX_<name>.parquet` with columns `['id', 'fold', <18 label probs>, 'pred', 'true']`.
- [ ] Experiment logged to `notes/progress.md` and `notes/experiments.jsonl` with 5-fold CV & Prior-Shift 5th-percentile score.
- [ ] If hypothesis succeeded: causal insight recorded in `notes/discoveries.md`.
- [ ] If hypothesis failed: failure mode and CV drop recorded in `notes/closed_ideas.md`.
- [ ] `notes/plan.md` updated with the next clear `[NEXT]` item.
- [ ] Ran verification: `python3 -m src.baton_check`.
