# Cookbook 04: Kaggle CLI & MCP Submission Pipeline

## Purpose
Deploys pre-trained model weights to Kaggle Datasets, pushes offline inference kernels, and validates + submits solutions.

## Step 1: Manage Pre-trained Weights via Kaggle Datasets
```bash
# Initialize dataset if not existing
kaggle datasets init -p artifacts/kaggle_dataset

# Version dataset with new weights
kaggle datasets version -p artifacts/kaggle_dataset -m "Update fold models" --dir-mode zip
```

## Step 2: Push Kaggle Inference Kernel
```bash
# Push notebook with code competition metadata
kaggle kernels push -p notebooks/
kaggle kernels status rajucode/name-that-llm-inference
kaggle kernels output rajucode/name-that-llm-inference -p artifacts/kernel_output/
```

## Step 3: Mandatory Pre-Submission Gates (Zero Wasted Quota Rule)
Before calling `competition_submit_code` or `kaggle competitions submit`, the Lead Orchestrator MUST verify:
1. **No Subagent Submissions:** Subagents must NEVER submit directly; they must stop after `kaggle kernels status` is `COMPLETE` and hand back to the Orchestrator.
2. **Hardware & Mount Verification via `GetKernel` + Commit Log:**
   - `_enable_gpu == True` and `_machine_shape == 'NvidiaTeslaT4'`
   - Commit log explicitly contains `CUDA=True, GPUs=2 (['Tesla T4', 'Tesla T4'])`
   - All `kernel_sources` and `dataset_sources` are public/accessible (never include 403/private kernels).
3. **Full Code-Path Execution on Commit:** Every model forward pass that runs on `len(test) >= 50` must also execute on a mini-batch during the interactive commit (`len(test) < 50`) so zero untested code paths exist on the hidden rerun.

## Step 4: Competition Submission via Python API or CLI
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
# IMPORTANT: file_name must be 'submission.parquet' (not a local directory path)
api.competition_submit_code(
    'submission.parquet',
    'CV: 0.XXXX | Model details',
    'name-that-llm',
    'rajucode/name-that-llm-exp002',
    4
)
```
