import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PROCESSED_DIR,
    EXPERIMENTS_LOG_PATH,
    FOLDS_PATH,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    LABELS,
    N_FOLDS,
    NUM_CLASSES,
    OOF_DIR,
    PROGRESS_PATH,
    RANDOM_SEED,
    ROOT_DIR,
    TRAIN_PATH,
)
from src.features.stylometry import StylometricFeatureExtractor
from src.validation import evaluate_predictions


def run_experiment_001() -> Tuple[pd.DataFrame, Dict]:
    """EXP-001: Multi-Granular TF-IDF + Stylometry + Fast Linear Ensemble (Ridge + SGD Huber + SGD Log)."""
    exp_id = "EXP-001"
    exp_name = "exp001_tfidf_linear"
    print(f"=== Starting {exp_id}: Multi-granular TF-IDF + Stylometry Fast Linear Baseline ===", flush=True)

    start_time = time.time()

    # 1. Load data and locked folds
    train_df = pd.read_csv(TRAIN_PATH)
    folds_df = pd.read_parquet(FOLDS_PATH)
    df = train_df.merge(folds_df[["id", "fold"]], on="id", how="left")
    assert df["fold"].notna().all(), "Some rows missing fold assignments!"

    y_labels = df["model"].values
    y_idx = np.array([LABEL_TO_IDX[lbl] for lbl in y_labels])
    prompt_ids = df["prompt_id"].values
    folds = df["fold"].values

    print(f"Dataset: {len(df)} rows across {len(np.unique(prompt_ids))} prompts and 5 folds.", flush=True)

    # 2. Extract or load cached Stylometric features
    sty_cache_path = DATA_PROCESSED_DIR / "train_stylometry.parquet"
    if sty_cache_path.exists():
        print(f"Loading cached stylometric features from {sty_cache_path}...", flush=True)
        sty_df = pd.read_parquet(sty_cache_path)
        X_sty_raw = sty_df.drop(columns=["id"]).values
    else:
        print("Extracting 165 stylometric features...", flush=True)
        sty_extractor = StylometricFeatureExtractor()
        X_sty_raw = sty_extractor.fit_transform(df["text"])
        col_names = sty_extractor.get_feature_names_out()
        cache_df = pd.DataFrame(X_sty_raw, columns=col_names)
        cache_df.insert(0, "id", df["id"])
        sty_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_parquet(sty_cache_path, index=False)

    print(f"Stylometric features ready: shape {X_sty_raw.shape}", flush=True)

    # 3. 5-Fold Cross-Validation Loop
    oof_probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float64)

    for fold in range(N_FOLDS):
        train_mask = folds != fold
        val_mask = folds == fold

        train_texts = df.loc[train_mask, "text"].values
        val_texts = df.loc[val_mask, "text"].values

        y_train = y_idx[train_mask]
        y_val = y_idx[val_mask]

        # Standardize stylometric features per fold to guarantee zero leakage
        scaler = StandardScaler()
        X_sty_tr = scaler.fit_transform(X_sty_raw[train_mask])
        X_sty_va = scaler.transform(X_sty_raw[val_mask])

        # Word TF-IDF (1-2 ngrams, 10k features)
        word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            min_df=2,
        )
        X_w_tr = word_vec.fit_transform(train_texts)
        X_w_va = word_vec.transform(val_texts)

        # Char TF-IDF (3-5 ngrams, 15k features)
        char_vec = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char",
            max_features=15000,
            sublinear_tf=True,
            min_df=3,
        )
        X_c_tr = char_vec.fit_transform(train_texts)
        X_c_va = char_vec.transform(val_texts)

        # Concatenate sparse TF-IDF and dense stylometry
        X_train_fold = hstack([X_w_tr, X_c_tr, X_sty_tr]).tocsr()
        X_val_fold = hstack([X_w_va, X_c_va, X_sty_va]).tocsr()

        # Model 1: Ridge Classifier with Softmax (alpha=1.5)
        ridge = RidgeClassifier(alpha=1.5, class_weight="balanced", random_state=RANDOM_SEED + fold)
        ridge.fit(X_train_fold, y_train)
        probs_ridge = softmax(ridge.decision_function(X_val_fold), axis=1)

        # Model 2: SGD Classifier (Modified Huber loss)
        clf_huber = SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-4,
            max_iter=250,
            class_weight="balanced",
            random_state=RANDOM_SEED + fold,
        )
        clf_huber.fit(X_train_fold, y_train)
        probs_huber = clf_huber.predict_proba(X_val_fold)

        # Model 3: SGD Classifier (Log loss)
        clf_sgd = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=250,
            class_weight="balanced",
            random_state=RANDOM_SEED + fold,
        )
        clf_sgd.fit(X_train_fold, y_train)
        probs_sgd = clf_sgd.predict_proba(X_val_fold)

        # Ensemble fold probabilities (Ridge 55% + Huber 25% + Log 20%)
        fold_probs = 0.55 * probs_ridge + 0.25 * probs_huber + 0.20 * probs_sgd
        oof_probs[val_mask] = fold_probs

        fold_acc = (np.argmax(oof_probs[val_mask], axis=1) == y_val).mean()
        print(f"Fold {fold} complete. Raw Accuracy: {fold_acc:.4f}", flush=True)

    elapsed = time.time() - start_time
    print(f"5-Fold CV training completed in {elapsed:.1f} seconds.", flush=True)

    # 4. Evaluate predictions
    metrics = evaluate_predictions(
        y_true=y_labels,
        oof_probs=oof_probs,
        prompt_ids=prompt_ids,
        folds=folds,
        run_stress_test=True,
    )

    # 5. Format and save OOF predictions
    oof_preds_labels = [IDX_TO_LABEL[idx] for idx in np.argmax(oof_probs, axis=1)]

    oof_df = pd.DataFrame({
        "id": df["id"],
        "fold": df["fold"],
        "pred": oof_preds_labels,
        "true": df["model"],
    })
    for idx, label in enumerate(LABELS):
        oof_df[label] = oof_probs[:, idx]

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = OOF_DIR / f"{exp_name}.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"OOF predictions saved to: {oof_path}", flush=True)

    # 6. Log to notes/experiments.jsonl
    log_entry = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Multi-granular TF-IDF (word 1-2, char 3-5) + 165 stylometric features + Fast Linear Ensemble (Ridge 55% + SGD Huber 25% + SGD Log 20%)",
        "cv_balanced_acc_mean": metrics["fold_mean"],
        "cv_balanced_acc_std": metrics["fold_std"],
        "overall_balanced_acc": metrics["overall_balanced_accuracy"],
        "prior_shift_p05": metrics["prior_shift_p05"],
        "prior_shift_mean": metrics["prior_shift_mean"],
        "provider_acc": metrics["provider_level_accuracy"],
        "worst_3_sibling_classes": metrics["worst_3_classes"],
        "oof_path": str(oof_path.relative_to(ROOT_DIR)),
        "runtime_seconds": round(elapsed, 1),
    }

    with open(EXPERIMENTS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    # 7. Update notes/progress.md table
    worst_str = ", ".join([f"{c} ({r:.2f})" for c, r in metrics["worst_3_classes"]])
    new_table_row = (
        f"| **{exp_id}** | {log_entry['description']} | "
        f"**{metrics['fold_mean']:.4f}** (±{metrics['fold_std']:.4f}) | "
        f"**{metrics['prior_shift_p05']:.4f}** | — | {worst_str} | "
        f"`{log_entry['oof_path']}` | Verified |\n"
    )

    progress_content = PROGRESS_PATH.read_text(encoding="utf-8")
    PROGRESS_PATH.write_text(progress_content + new_table_row, encoding="utf-8")
    print("Updated notes/progress.md scoreboard.", flush=True)

    return oof_df, metrics


if __name__ == "__main__":
    run_experiment_001()
