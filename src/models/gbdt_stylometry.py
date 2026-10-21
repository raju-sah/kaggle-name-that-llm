import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
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


def run_experiment_002() -> Tuple[pd.DataFrame, Dict]:
    """EXP-002: Dense Stylometry + TF-IDF SVD & Ridge Projections + LightGBM GBDT."""
    exp_id = "EXP-002"
    exp_name = "exp002_gbdt_stylometry"
    print(f"=== Starting {exp_id}: Dense Stylometry + SVD/Ridge Projections + LightGBM GBDT ===")

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

    print(f"Dataset: {len(df)} rows across {len(np.unique(prompt_ids))} prompts and 5 folds.")

    # 2. Extract Stylometric features
    print("Extracting stylometric features...")
    sty_extractor = StylometricFeatureExtractor()
    X_sty_raw = sty_extractor.fit_transform(df["text"])
    print(f"Stylometric features extracted: shape {X_sty_raw.shape}")

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

        # Word TF-IDF (1-3 ngrams)
        word_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20000,
            sublinear_tf=True,
            min_df=2,
        )
        X_w_tr = word_vec.fit_transform(train_texts)
        X_w_va = word_vec.transform(val_texts)

        # Char TF-IDF (2-5 ngrams)
        char_vec = TfidfVectorizer(
            ngram_range=(2, 5),
            analyzer="char",
            max_features=25000,
            sublinear_tf=True,
            min_df=3,
        )
        X_c_tr = char_vec.fit_transform(train_texts)
        X_c_va = char_vec.transform(val_texts)

        # Char-WB TF-IDF (3-5 ngrams)
        char_wb_vec = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char_wb",
            max_features=20000,
            sublinear_tf=True,
            min_df=3,
        )
        X_cwb_tr = char_wb_vec.fit_transform(train_texts)
        X_cwb_va = char_wb_vec.transform(val_texts)

        # Concatenate sparse TF-IDF
        X_tfidf_tr = hstack([X_w_tr, X_c_tr, X_cwb_tr]).tocsr()
        X_tfidf_va = hstack([X_w_va, X_c_va, X_cwb_va]).tocsr()

        # Dimensionality Reduction: TruncatedSVD (64 components)
        svd = TruncatedSVD(n_components=64, random_state=RANDOM_SEED + fold)
        X_svd_tr = svd.fit_transform(X_tfidf_tr)
        X_svd_va = svd.transform(X_tfidf_va)

        # Supervised Ridge Margin Projections (18 classes)
        ridge = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=RANDOM_SEED + fold)
        ridge.fit(X_tfidf_tr, y_train)
        X_ridge_tr = ridge.decision_function(X_tfidf_tr)
        X_ridge_va = ridge.decision_function(X_tfidf_va)

        # Combine into dense feature matrix
        X_train_dense = np.hstack([X_sty_tr, X_svd_tr, X_ridge_tr])
        X_val_dense = np.hstack([X_sty_va, X_svd_va, X_ridge_va])

        # LightGBM Multi-class Classifier
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=NUM_CLASSES,
            class_weight="balanced",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train_dense, y_train)
        probs = clf.predict_proba(X_val_dense)
        oof_probs[val_mask] = probs

        fold_acc = (np.argmax(probs, axis=1) == y_val).mean()
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
        "description": "Dense Stylometry (165) + TF-IDF SVD (64) + Ridge Margins (18) + LightGBM GBDT (250 trees)",
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
    run_experiment_002()
