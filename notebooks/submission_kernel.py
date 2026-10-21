"""Name That LLM - EXP-001 Baseline Inference & Submission Kernel
Multi-granular TF-IDF + 165 Stylometric Features + Calibrated Linear Ensemble (Ridge + SGD Huber + SGD Log)
"""

import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.preprocessing import StandardScaler

# 18 Canonical Models
LABELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# Stylometry constants & regex
SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+(?:\s+|$)")
WORD_SPLIT_REGEX = re.compile(r"\b\w+\b")
VOWEL_REGEX = re.compile(r"[aeiouyAEIOUY]")
CONSONANT_REGEX = re.compile(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]")

DISCOURSE_MARKERS = [
    "however", "moreover", "furthermore", "crucially", "additionally",
    "consequently", "therefore", "thus", "specifically", "notably",
    "importantly", "in particular", "for instance", "for example",
    "in contrast", "on the other hand", "alternatively", "conversely",
    "meanwhile", "ultimately", "essentially", "fundamentally", "overall",
    "in summary", "to summarize", "in conclusion", "first", "second",
    "third", "finally", "next", "indeed", "certainly", "clearly",
    "obviously", "arguably", "perhaps", "likely", "although", "though",
    "whereas", "while", "despite", "regardless", "because", "since",
    "due to", "rather", "instead", "namely", "in fact", "also",
    "likewise", "similarly", "delve", "testament", "tapestry", "beacon",
    "foster", "underscore", "pivotal", "nuanced", "holistic",
    "multifaceted", "intertwined",
]


def estimate_syllables(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and len(word) > 2 and not word.endswith("ee"):
        count -= 1
    return max(1, count)


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_: List[str] = []

    def fit(self, X: Union[pd.Series, List[str], np.ndarray], y=None):
        dummy_feats = self._extract_text_features("Sample passage for initialization.")
        self.feature_names_ = list(dummy_feats.keys())
        return self

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.Series) or isinstance(X, np.ndarray):
            texts = X.tolist()
        else:
            texts = list(X)
        feature_dicts = [self._extract_text_features(t) for t in texts]
        matrix = np.array([[fd[col] for col in self.feature_names_] for fd in feature_dicts], dtype=np.float32)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

    def _extract_text_features(self, text: str) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        text_len = len(text)
        text_lower = text.lower()

        raw_words = text.split()
        words = WORD_SPLIT_REGEX.findall(text)
        word_count = len(words)
        clean_words_lower = [w.lower() for w in words]

        sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text.strip()) if s.strip()]
        sent_count = max(1, len(sentences))

        feats["char_count"] = float(text_len)
        feats["word_count"] = float(word_count)
        feats["sent_count"] = float(sent_count)
        feats["raw_word_count"] = float(len(raw_words))

        words_per_sent = [len(WORD_SPLIT_REGEX.findall(s)) for s in sentences] if sentences else [word_count]
        wps_arr = np.array(words_per_sent, dtype=np.float32)
        mean_wps = float(np.mean(wps_arr))
        std_wps = float(np.std(wps_arr))
        feats["avg_words_per_sent"] = mean_wps
        feats["std_words_per_sent"] = std_wps
        feats["min_words_per_sent"] = float(np.min(wps_arr))
        feats["max_words_per_sent"] = float(np.max(wps_arr))
        feats["burstiness_cv"] = std_wps / (mean_wps + 1e-5)

        word_lens = [len(w) for w in words] if words else [0]
        wl_arr = np.array(word_lens, dtype=np.float32)
        feats["avg_chars_per_word"] = float(np.mean(wl_arr))
        feats["std_chars_per_word"] = float(np.std(wl_arr))
        feats["word_len_p25"] = float(np.percentile(wl_arr, 25))
        feats["word_len_p50"] = float(np.percentile(wl_arr, 50))
        feats["word_len_p75"] = float(np.percentile(wl_arr, 75))
        feats["word_len_p90"] = float(np.percentile(wl_arr, 90))
        feats["short_word_ratio"] = float(np.mean(wl_arr <= 3))
        feats["long_word_ratio"] = float(np.mean(wl_arr >= 8))

        chars_denom = max(1.0, float(text_len)) / 1000.0
        punct_symbols = {
            "em_dash": "—", "en_dash": "–", "hyphen": "-", "double_hyphen": "--",
            "semicolon": ";", "colon": ":", "comma": ",", "period": ".",
            "exclamation": "!", "question": "?", "single_quote": "'", "double_quote": '"',
            "curly_sq_open": "‘", "curly_sq_close": "’", "curly_dq_open": "“",
            "curly_dq_close": "”", "lparen": "(", "rparen": ")", "lbracket": "[",
            "rbracket": "]", "slash": "/", "backslash": "\\", "asterisk": "*",
            "backtick": "`", "underscore": "_", "ellipsis_dots": "...",
            "ellipsis_char": "…", "ampersand": "&", "percent": "%",
            "dollar": "$", "hash": "#",
        }

        total_punct = 0
        for name, sym in punct_symbols.items():
            cnt = text.count(sym)
            feats[f"cnt_{name}"] = float(cnt)
            feats[f"rate_{name}"] = float(cnt) / chars_denom
            total_punct += cnt

        feats["total_punct_count"] = float(total_punct)
        feats["punct_density"] = float(total_punct) / max(1.0, float(text_len))
        feats["em_dash_to_comma"] = float(feats["cnt_em_dash"]) / (feats["cnt_comma"] + 1e-4)
        feats["semicolon_to_colon"] = float(feats["cnt_semicolon"]) / (feats["cnt_colon"] + 1e-4)
        feats["paren_count"] = feats["cnt_lparen"] + feats["cnt_rparen"]
        feats["bracket_count"] = feats["cnt_lbracket"] + feats["cnt_rbracket"]
        feats["quotes_total"] = (
            feats["cnt_single_quote"] + feats["cnt_double_quote"] +
            feats["cnt_curly_sq_open"] + feats["cnt_curly_sq_close"] +
            feats["cnt_curly_dq_open"] + feats["cnt_curly_dq_close"]
        )

        words_denom = max(1.0, float(word_count))
        unique_words = set(clean_words_lower)
        feats["ttr"] = float(len(unique_words)) / words_denom

        word_counts = Counter(clean_words_lower)
        hapax_count = sum(1 for c in word_counts.values() if c == 1)
        feats["hapax_legomena_ratio"] = float(hapax_count) / words_denom

        n_upper = sum(1 for c in text if c.isupper())
        n_digits = sum(1 for c in text if c.isdigit())
        n_spaces = sum(1 for c in text if c.isspace())
        n_vowels = len(VOWEL_REGEX.findall(text))
        n_consonants = len(CONSONANT_REGEX.findall(text))
        n_ascii = sum(1 for c in text if ord(c) < 128)

        feats["upper_ratio"] = float(n_upper) / max(1.0, float(text_len))
        feats["digit_ratio"] = float(n_digits) / max(1.0, float(text_len))
        feats["space_ratio"] = float(n_spaces) / max(1.0, float(text_len))
        feats["vowel_ratio"] = float(n_vowels) / max(1.0, float(text_len))
        feats["consonant_ratio"] = float(n_consonants) / max(1.0, float(text_len))
        feats["vowel_consonant_ratio"] = float(n_vowels) / (float(n_consonants) + 1e-4)
        feats["ascii_ratio"] = float(n_ascii) / max(1.0, float(text_len))
        feats["non_ascii_ratio"] = 1.0 - feats["ascii_ratio"]

        n_title = sum(1 for w in words if w.istitle())
        feats["titlecase_ratio"] = float(n_title) / words_denom

        syllables = [estimate_syllables(w) for w in clean_words_lower] if clean_words_lower else [1]
        total_syllables = sum(syllables)
        avg_syllables = float(total_syllables) / words_denom
        feats["avg_syllables_per_word"] = avg_syllables

        flesch_ease = 206.835 - 1.015 * mean_wps - 84.6 * avg_syllables
        feats["flesch_reading_ease"] = float(np.clip(flesch_ease, -50.0, 150.0))
        kincaid_grade = 0.39 * mean_wps + 11.8 * avg_syllables - 15.59
        feats["kincaid_grade_level"] = float(np.clip(kincaid_grade, 0.0, 30.0))

        words_per_1000 = words_denom / 1000.0
        for marker in DISCOURSE_MARKERS:
            m_cnt = len(re.findall(r"\b" + re.escape(marker) + r"\b", text_lower))
            feats[f"marker_{marker.replace(' ', '_')}"] = float(m_cnt) / words_per_1000

        return feats


def sinkhorn_marginal_balancing(probs: np.ndarray, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
    """Computes column multipliers to balance predictions towards uniform marginals."""
    n, k = probs.shape
    if n < k:
        return probs

    target_col = np.ones(k, dtype=np.float64) / k
    target_row = np.ones(n, dtype=np.float64) / n

    P = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    P = P / P.sum()

    u = np.ones(n, dtype=np.float64) / n
    v = np.ones(k, dtype=np.float64) / k

    for _ in range(max_iter):
        u_prev = u
        u = target_row / (P @ v + 1e-12)
        v = target_col / (P.T @ u + 1e-12)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    scaled_probs = P * v[np.newaxis, :]
    return scaled_probs


def find_competition_files() -> Tuple[Path, Path]:
    candidates = [
        Path("/kaggle/input/competitions/name-that-llm"),
        Path("/kaggle/input/name-that-llm"),
        Path("data/raw"),
        Path("../data/raw"),
    ]
    for c in candidates:
        tr = c / "train.csv"
        te = c / "test.csv"
        if tr.exists() and te.exists():
            return tr, te

    # Dynamic search under /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for tr in kaggle_input.rglob("train.csv"):
            te = tr.parent / "test.csv"
            if te.exists():
                return tr, te

    raise FileNotFoundError("Could not find train.csv and test.csv in search paths.")


def main():
    start_time = time.time()
    print("=== Name That LLM: EXP-001 Inference Pipeline Starting ===", flush=True)

    train_path, test_path = find_competition_files()
    print(f"Loading data from:\n  Train: {train_path}\n  Test:  {test_path}", flush=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}", flush=True)

    assert "text" in train_df.columns and "model" in train_df.columns
    assert "text" in test_df.columns and "id" in test_df.columns

    y_train_labels = train_df["model"].values
    y_train = np.array([LABEL_TO_IDX[lbl] for lbl in y_train_labels])

    # 1. Stylometric Features
    print("Extracting stylometric features...", flush=True)
    sty_extractor = StylometricFeatureExtractor()
    X_sty_train_raw = sty_extractor.fit_transform(train_df["text"])
    X_sty_test_raw = sty_extractor.transform(test_df["text"])

    scaler = StandardScaler()
    X_sty_tr = scaler.fit_transform(X_sty_train_raw)
    X_sty_te = scaler.transform(X_sty_test_raw)
    print(f"Stylometric features ready: train shape {X_sty_tr.shape}, test shape {X_sty_te.shape}", flush=True)

    # 2. Word TF-IDF
    print("Extracting Word TF-IDF...", flush=True)
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
    )
    X_w_tr = word_vec.fit_transform(train_df["text"])
    X_w_te = word_vec.transform(test_df["text"])

    # 3. Char TF-IDF
    print("Extracting Char TF-IDF...", flush=True)
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char",
        max_features=15000,
        sublinear_tf=True,
        min_df=3,
    )
    X_c_tr = char_vec.fit_transform(train_df["text"])
    X_c_te = char_vec.transform(test_df["text"])

    # 4. Feature Stacking
    X_train = hstack([X_w_tr, X_c_tr, X_sty_tr]).tocsr()
    X_test = hstack([X_w_te, X_c_te, X_sty_te]).tocsr()
    print(f"Combined feature shapes: train {X_train.shape}, test {X_test.shape}", flush=True)

    # 5. Fit Linear Ensemble
    print("Training Ridge Classifier...", flush=True)
    ridge = RidgeClassifier(alpha=1.5, class_weight="balanced", random_state=42)
    ridge.fit(X_train, y_train)
    probs_ridge = softmax(ridge.decision_function(X_test), axis=1)

    print("Training SGD Classifier (modified huber)...", flush=True)
    clf_huber = SGDClassifier(
        loss="modified_huber",
        penalty="l2",
        alpha=1e-4,
        max_iter=250,
        class_weight="balanced",
        random_state=42,
    )
    clf_huber.fit(X_train, y_train)
    probs_huber = clf_huber.predict_proba(X_test)

    print("Training SGD Classifier (log loss)...", flush=True)
    clf_sgd = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=250,
        class_weight="balanced",
        random_state=42,
    )
    clf_sgd.fit(X_train, y_train)
    probs_sgd = clf_sgd.predict_proba(X_test)

    # Ensemble blending
    ensemble_probs = 0.55 * probs_ridge + 0.25 * probs_huber + 0.20 * probs_sgd

    # 6. Prior-Shift Calibration / Sinkhorn Balancing
    if len(test_df) >= NUM_CLASSES:
        print("Applying Sinkhorn marginal balancing for test set...", flush=True)
        balanced_probs = sinkhorn_marginal_balancing(ensemble_probs)
        pred_indices = np.argmax(balanced_probs, axis=1)

        # Anti-starvation check: ensure no class has 0 predictions when N >= 18
        pred_counts = Counter(pred_indices)
        for cls_idx in range(NUM_CLASSES):
            if pred_counts[cls_idx] == 0:
                # Reassign the test row that is most confident for this class
                best_row = int(np.argmax(balanced_probs[:, cls_idx]))
                print(f"Anti-starvation reallocation: class {IDX_TO_LABEL[cls_idx]} assigned to row {best_row}", flush=True)
                pred_indices[best_row] = cls_idx
                pred_counts[cls_idx] += 1
    else:
        print(f"Test size ({len(test_df)}) < 18, applying direct argmax.", flush=True)
        pred_indices = np.argmax(ensemble_probs, axis=1)

    pred_labels = [IDX_TO_LABEL[idx] for idx in pred_indices]

    # 7. Construct & Validate Submission
    sub_df = pd.DataFrame({
        "id": test_df["id"],
        "model": pred_labels,
    })

    # Invariant Validations
    assert len(sub_df) == len(test_df), f"Row count mismatch: {len(sub_df)} vs {len(test_df)}"
    assert list(sub_df.columns) == ["id", "model"], f"Invalid columns: {list(sub_df.columns)}"
    assert sub_df["id"].notna().all(), "Null IDs found in submission!"
    assert sub_df["model"].notna().all(), "Null model predictions found in submission!"
    assert sub_df["id"].nunique() == len(sub_df), "Duplicate IDs found in submission!"
    invalid_labels = set(sub_df["model"]) - set(LABELS)
    assert not invalid_labels, f"Invalid model labels predicted: {invalid_labels}"

    if len(test_df) >= NUM_CLASSES:
        assert sub_df["model"].nunique() == NUM_CLASSES, (
            f"Expected all {NUM_CLASSES} classes to be predicted, but got {sub_df['model'].nunique()}"
        )
        print(f"Validated: All {NUM_CLASSES} classes are represented in predictions.", flush=True)

    # 8. Save submission.parquet
    out_dir = Path("/kaggle/working")
    if not out_dir.exists():
        out_dir = Path(".")

    out_path = out_dir / "submission.parquet"
    sub_df.to_parquet(out_path, index=False)
    print(f"Successfully generated submission at: {out_path}", flush=True)
    print(f"File size: {out_path.stat().st_size} bytes", flush=True)
    print(f"Head of submission:\n{sub_df.head(10)}", flush=True)

    # Also save to current dir if different
    curr_sub = Path("submission.parquet")
    if out_path.resolve() != curr_sub.resolve():
        sub_df.to_parquet(curr_sub, index=False)

    elapsed = time.time() - start_time
    print(f"Inference pipeline completed successfully in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
