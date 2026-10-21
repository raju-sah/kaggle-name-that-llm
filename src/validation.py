import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    FOLDS_PATH,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    LABELS,
    N_FOLDS,
    NUM_CLASSES,
    PROVIDER_MAP,
    RANDOM_SEED,
    TEST_PATH,
    TRAIN_PATH,
)


def create_locked_folds(
    train_path: Path = TRAIN_PATH,
    folds_path: Path = FOLDS_PATH,
    n_splits: int = N_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Creates and saves locked StratifiedGroupKFold splits by prompt_id."""
    df = pd.read_csv(train_path)
    print(f"Loaded train dataset: {len(df)} rows across {df['prompt_id'].nunique()} prompts.")

    # Sort to ensure deterministic split across environments
    df = df.sort_values(by=["id"]).reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    df["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(df, y=df["model"], groups=df["prompt_id"])
    ):
        df.loc[val_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "Unassigned folds detected!"

    # Verify fold integrity
    verify_fold_integrity(df, n_splits=n_splits)

    # Save to parquet
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "prompt_id", "model", "fold"]].to_parquet(folds_path, index=False)
    print(f"Successfully created and saved locked folds to: {folds_path}")
    return df


def verify_fold_integrity(df: pd.DataFrame, n_splits: int = N_FOLDS) -> None:
    """Verifies zero prompt leakage and representation of all 18 classes per fold."""
    for fold in range(n_splits):
        val_prompts = set(df[df["fold"] == fold]["prompt_id"])
        train_prompts = set(df[df["fold"] != fold]["prompt_id"])
        overlap = val_prompts.intersection(train_prompts)
        assert len(overlap) == 0, f"Prompt leakage detected in fold {fold}: {len(overlap)} shared prompts!"

        val_classes = set(df[df["fold"] == fold]["model"])
        missing_classes = set(LABELS) - val_classes
        assert (
            len(missing_classes) == 0
        ), f"Missing classes in fold {fold}: {missing_classes}"

    print(f"Integrity check PASSED: Zero prompt overlap and all 18 classes represented across {n_splits} folds.")


def dirichlet_prior_shift_stress_test(
    y_true_indices: np.ndarray,
    oof_probs: np.ndarray,
    n_simulations: int = 1000,
    test_size: int = 4000,
    alpha: float = 1.0,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """Evaluates Balanced Accuracy across n_simulations non-uniform test distributions

    simulated from Dirichlet(alpha).
    """
    rng = np.random.default_rng(random_seed)
    y_preds = np.argmax(oof_probs, axis=1)

    # Group sample indices by true class
    class_indices = [np.where(y_true_indices == k)[0] for k in range(NUM_CLASSES)]
    for k in range(NUM_CLASSES):
        assert len(class_indices[k]) > 0, f"Class {k} has no instances in validation set!"

    sim_scores = np.empty(n_simulations, dtype=np.float64)

    for i in range(n_simulations):
        # Sample non-uniform prior
        priors = rng.dirichlet(np.ones(NUM_CLASSES) * alpha)
        # Allocate sample counts per class
        counts = rng.multinomial(test_size, priors)
        # Ensure at least 1 sample per class to avoid division by zero
        counts = np.maximum(counts, 1)

        sim_true = []
        sim_pred = []
        for k in range(NUM_CLASSES):
            sampled_idx = rng.choice(class_indices[k], size=counts[k], replace=True)
            sim_true.append(y_true_indices[sampled_idx])
            sim_pred.append(y_preds[sampled_idx])

        y_test_sim = np.concatenate(sim_true)
        y_pred_sim = np.concatenate(sim_pred)
        sim_scores[i] = balanced_accuracy_score(y_test_sim, y_pred_sim)

    return {
        "prior_shift_mean": float(np.mean(sim_scores)),
        "prior_shift_std": float(np.std(sim_scores)),
        "prior_shift_p05": float(np.percentile(sim_scores, 5)),
        "prior_shift_p50": float(np.percentile(sim_scores, 50)),
        "prior_shift_p95": float(np.percentile(sim_scores, 95)),
    }


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray, List[str]],
    oof_probs: np.ndarray,
    prompt_ids: Optional[Union[pd.Series, np.ndarray]] = None,
    folds: Optional[Union[pd.Series, np.ndarray]] = None,
    run_stress_test: bool = True,
) -> Dict[str, Any]:
    """Comprehensive evaluation returning overall balanced accuracy, fold scores,

    per-class recalls, provider confusion, and prior-shift stress test.
    """
    # Convert y_true to integer indices if string
    if isinstance(y_true, pd.Series) or isinstance(y_true, list):
        y_true_arr = np.array(y_true)
    else:
        y_true_arr = y_true

    if isinstance(y_true_arr[0], str):
        y_true_idx = np.array([LABEL_TO_IDX[lbl] for lbl in y_true_arr])
    else:
        y_true_idx = y_true_arr.astype(int)

    oof_preds = np.argmax(oof_probs, axis=1)
    overall_balanced_acc = float(balanced_accuracy_score(y_true_idx, oof_preds))

    # Per-class recall
    cm = confusion_matrix(y_true_idx, oof_preds, labels=list(range(NUM_CLASSES)))
    per_class_recalls: Dict[str, float] = {}
    for k in range(NUM_CLASSES):
        total_k = cm[k, :].sum()
        recall_k = float(cm[k, k] / total_k) if total_k > 0 else 0.0
        per_class_recalls[IDX_TO_LABEL[k]] = recall_k

    # Identify worst 3 sibling classes
    sorted_recalls = sorted(per_class_recalls.items(), key=lambda x: x[1])
    worst_3_recalls = sorted_recalls[:3]

    # Provider-level accuracy and confusion
    true_providers = np.array([PROVIDER_MAP[IDX_TO_LABEL[k]] for k in y_true_idx])
    pred_providers = np.array([PROVIDER_MAP[IDX_TO_LABEL[k]] for k in oof_preds])
    provider_acc = float(np.mean(true_providers == pred_providers))

    # Fold scores
    fold_scores: List[float] = []
    if folds is not None:
        folds_arr = np.array(folds)
        unique_folds = sorted(np.unique(folds_arr))
        for f in unique_folds:
            mask = folds_arr == f
            f_score = float(balanced_accuracy_score(y_true_idx[mask], oof_preds[mask]))
            fold_scores.append(f_score)

    results: Dict[str, Any] = {
        "overall_balanced_accuracy": overall_balanced_acc,
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean(fold_scores)) if fold_scores else overall_balanced_acc,
        "fold_std": float(np.std(fold_scores)) if fold_scores else 0.0,
        "provider_level_accuracy": provider_acc,
        "per_class_recalls": per_class_recalls,
        "worst_3_classes": worst_3_recalls,
    }

    if run_stress_test:
        stress_results = dirichlet_prior_shift_stress_test(y_true_idx, oof_probs)
        results.update(stress_results)

    # Print summary report
    print("=" * 60)
    print(f"5-Fold Balanced Accuracy: {results['fold_mean']:.4f} (std: {results['fold_std']:.4f})")
    print(f"Overall Balanced Accuracy: {results['overall_balanced_accuracy']:.4f}")
    print(f"Provider Classification Accuracy: {results['provider_level_accuracy']:.4f}")
    if run_stress_test:
        print(f"Dirichlet Prior-Shift 5th-percentile (p05): {results['prior_shift_p05']:.4f}")
        print(f"Dirichlet Prior-Shift Mean: {results['prior_shift_mean']:.4f}")
    print("-" * 60)
    print("Worst 3 Sibling Classes:")
    for lbl, rec in worst_3_recalls:
        print(f"  - {lbl} ({PROVIDER_MAP[lbl]}): {rec:.4f} recall")
    print("=" * 60)

    return results


def validate_submission_parquet(
    sub_path: Union[str, Path],
    test_path: Union[str, Path] = TEST_PATH,
) -> Tuple[bool, List[str]]:
    """Strictly validates submission.parquet against all competition constraints."""
    sub_path = Path(sub_path)
    test_path = Path(test_path)
    errors: List[str] = []

    if not sub_path.exists():
        return False, [f"File does not exist: {sub_path}"]

    try:
        sub = pd.read_parquet(sub_path)
    except Exception as e:
        return False, [f"Failed to read Parquet file: {e}"]

    expected_cols = ["id", "model"]
    if list(sub.columns) != expected_cols:
        errors.append(f"Columns mismatch: expected {expected_cols}, got {list(sub.columns)}")

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        if len(sub) != len(test_df):
            errors.append(f"Row count mismatch: expected {len(test_df)}, got {len(sub)}")
        if not (sub["id"].values == test_df["id"].values).all():
            errors.append("IDs do not match test.csv exactly in sequence!")

    if sub["id"].duplicated().any():
        errors.append(f"Found {sub['id'].duplicated().sum()} duplicate IDs!")

    if sub["model"].isna().any():
        errors.append(f"Found {sub['model'].isna().sum()} null values in 'model' column!")

    invalid_labels = set(sub["model"]) - set(LABELS)
    if len(invalid_labels) > 0:
        errors.append(f"Invalid model labels found: {invalid_labels}")

    # Check for starved classes
    counts = sub["model"].value_counts()
    missing_from_preds = set(LABELS) - set(counts.index)
    if missing_from_preds:
        print(f"[WARNING] Starved classes (0 predictions): {missing_from_preds}")

    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV and Submission Validation Tool")
    parser.add_argument("--verify-folds", action="store_true", help="Generate or verify locked CV folds")
    parser.add_argument("--check-submission", type=str, default=None, help="Path to submission.parquet to validate")
    args = parser.parse_args()

    if args.verify_folds:
        if FOLDS_PATH.exists():
            print(f"Loading existing folds from: {FOLDS_PATH}")
            folds_df = pd.read_parquet(FOLDS_PATH)
            verify_fold_integrity(folds_df)
        else:
            print(f"Generating new locked folds at: {FOLDS_PATH}")
            create_locked_folds()

    if args.check_submission:
        valid, errs = validate_submission_parquet(args.check_submission)
        if valid:
            print(f"[SUCCESS] Submission {args.check_submission} passed all validation checks!")
        else:
            print(f"[FAILED] Submission {args.check_submission} has errors:")
            for err in errs:
                print(f"  - {err}")
            sys.exit(1)
