# Cookbook 01: Locked CV & Metric Evaluation

## Purpose
Standardizes out-of-fold cross-validation and evaluation across all experiments to guarantee 100% zero prompt leakage and strict reproducibility.

## How to Load Locked Folds
```python
import pandas as pd
from src.config import FOLDS_PATH, TRAIN_PATH

# Load train text and merge with locked folds
train_df = pd.read_csv(TRAIN_PATH)
folds_df = pd.read_parquet(FOLDS_PATH)
train_df = train_df.merge(folds_df[['id', 'fold']], on='id', how='left')
assert train_df['fold'].notna().all(), "Missing fold assignments!"
```

## How to Evaluate and Log Predictions
```python
import numpy as np
from src.validation import evaluate_predictions

# oof_probs is a (2699, 18) numpy array of predicted probabilities
# columns must match src.config.LABELS in exact order
metrics = evaluate_predictions(
    y_true=train_df['model'],
    oof_probs=oof_probs,
    prompt_ids=train_df['prompt_id'],
    folds=train_df['fold'],
    run_stress_test=True,
)

print(f"CV Balanced Accuracy: {metrics['fold_mean']:.4f} +/- {metrics['fold_std']:.4f}")
print(f"Dirichlet Prior-Shift p05: {metrics['prior_shift_p05']:.4f}")
```

## Mandatory OOF Parquet Schema
Save all out-of-fold prediction matrices to `artifacts/oof/<exp_id>_<name>.parquet`:
- `id`: passage id (str)
- `fold`: validation fold index (0..4)
- `pred`: argmax predicted model string (str)
- `true`: ground truth model string (str)
- 18 columns named after each model in `src.config.LABELS` containing class probabilities.
