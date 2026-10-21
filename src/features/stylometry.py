import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Pre-compiled regex patterns for speed
SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+(?:\s+|$)")
WORD_SPLIT_REGEX = re.compile(r"\b\w+\b")
VOWEL_REGEX = re.compile(r"[aeiouyAEIOUY]")
CONSONANT_REGEX = re.compile(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]")

# Curated list of discourse markers and distinctive stylistic tokens (60+)
DISCOURSE_MARKERS = [
    "however",
    "moreover",
    "furthermore",
    "crucially",
    "additionally",
    "consequently",
    "therefore",
    "thus",
    "specifically",
    "notably",
    "importantly",
    "in particular",
    "for instance",
    "for example",
    "in contrast",
    "on the other hand",
    "alternatively",
    "conversely",
    "meanwhile",
    "ultimately",
    "essentially",
    "fundamentally",
    "overall",
    "in summary",
    "to summarize",
    "in conclusion",
    "first",
    "second",
    "third",
    "finally",
    "next",
    "indeed",
    "certainly",
    "clearly",
    "obviously",
    "arguably",
    "perhaps",
    "likely",
    "although",
    "though",
    "whereas",
    "while",
    "despite",
    "regardless",
    "because",
    "since",
    "due to",
    "rather",
    "instead",
    "namely",
    "in fact",
    "also",
    "likewise",
    "similarly",
    # Lab and RLHF-distinctive keywords
    "delve",
    "testament",
    "tapestry",
    "beacon",
    "foster",
    "underscore",
    "pivotal",
    "nuanced",
    "holistic",
    "multifaceted",
    "intertwined",
]


def estimate_syllables(word: str) -> int:
    """Fast rule-based syllable estimator."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and len(word) > 2 and not word.endswith("ee"):
        count -= 1
    return max(1, count)


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """Zero-leakage, deterministic 120+ feature extractor capturing punctuation,

    syntax rhythm, lexical diversity, and discourse markers.
    """

    def __init__(self):
        self.feature_names_: List[str] = []

    def fit(self, X: Union[pd.Series, List[str], np.ndarray], y=None):
        # Deterministic, stateless transformer
        dummy_feats = self._extract_text_features("Sample passage for initialization.")
        self.feature_names_ = list(dummy_feats.keys())
        return self

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.tolist()
        else:
            texts = list(X)

        feature_dicts = [self._extract_text_features(t) for t in texts]
        matrix = np.array([[fd[col] for col in self.feature_names_] for fd in feature_dicts], dtype=np.float32)
        # Replace any residual NaNs/Infs with 0.0
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

    def _extract_text_features(self, text: str) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        text_len = len(text)
        text_lower = text.lower()

        # 1. Basic Tokenization
        raw_words = text.split()
        words = WORD_SPLIT_REGEX.findall(text)
        word_count = len(words)
        clean_words_lower = [w.lower() for w in words]

        sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text.strip()) if s.strip()]
        sent_count = max(1, len(sentences))

        # Length & Counts
        feats["char_count"] = float(text_len)
        feats["word_count"] = float(word_count)
        feats["sent_count"] = float(sent_count)
        feats["raw_word_count"] = float(len(raw_words))

        # Sentence rhythm & burstiness
        words_per_sent = [len(WORD_SPLIT_REGEX.findall(s)) for s in sentences] if sentences else [word_count]
        wps_arr = np.array(words_per_sent, dtype=np.float32)
        mean_wps = float(np.mean(wps_arr))
        std_wps = float(np.std(wps_arr))
        feats["avg_words_per_sent"] = mean_wps
        feats["std_words_per_sent"] = std_wps
        feats["min_words_per_sent"] = float(np.min(wps_arr))
        feats["max_words_per_sent"] = float(np.max(wps_arr))
        feats["burstiness_cv"] = std_wps / (mean_wps + 1e-5)

        # Word lengths
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

        # 2. Punctuation Forensics (Raw + Normalized per 1000 characters)
        chars_denom = max(1.0, float(text_len)) / 1000.0

        punct_symbols = {
            "em_dash": "—",
            "en_dash": "–",
            "hyphen": "-",
            "double_hyphen": "--",
            "semicolon": ";",
            "colon": ":",
            "comma": ",",
            "period": ".",
            "exclamation": "!",
            "question": "?",
            "single_quote": "'",
            "double_quote": '"',
            "curly_sq_open": "‘",
            "curly_sq_close": "’",
            "curly_dq_open": "“",
            "curly_dq_close": "”",
            "lparen": "(",
            "rparen": ")",
            "lbracket": "[",
            "rbracket": "]",
            "slash": "/",
            "backslash": "\\",
            "asterisk": "*",
            "backtick": "`",
            "underscore": "_",
            "ellipsis_dots": "...",
            "ellipsis_char": "…",
            "ampersand": "&",
            "percent": "%",
            "dollar": "$",
            "hash": "#",
        }

        total_punct = 0
        for name, sym in punct_symbols.items():
            cnt = text.count(sym)
            feats[f"cnt_{name}"] = float(cnt)
            feats[f"rate_{name}"] = float(cnt) / chars_denom
            total_punct += cnt

        feats["total_punct_count"] = float(total_punct)
        feats["punct_density"] = float(total_punct) / max(1.0, float(text_len))

        # Punctuation ratios
        feats["em_dash_to_comma"] = float(feats["cnt_em_dash"]) / (feats["cnt_comma"] + 1e-4)
        feats["semicolon_to_colon"] = float(feats["cnt_semicolon"]) / (feats["cnt_colon"] + 1e-4)
        feats["paren_count"] = feats["cnt_lparen"] + feats["cnt_rparen"]
        feats["bracket_count"] = feats["cnt_lbracket"] + feats["cnt_rbracket"]
        feats["quotes_total"] = (
            feats["cnt_single_quote"]
            + feats["cnt_double_quote"]
            + feats["cnt_curly_sq_open"]
            + feats["cnt_curly_sq_close"]
            + feats["cnt_curly_dq_open"]
            + feats["cnt_curly_dq_close"]
        )

        # 3. Lexical Diversity & Character Statistics
        words_denom = max(1.0, float(word_count))
        unique_words = set(clean_words_lower)
        feats["ttr"] = float(len(unique_words)) / words_denom

        word_counts = Counter(clean_words_lower)
        hapax_count = sum(1 for c in word_counts.values() if c == 1)
        feats["hapax_legomena_ratio"] = float(hapax_count) / words_denom

        # Character classes
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

        # Titlecase words
        n_title = sum(1 for w in words if w.istitle())
        feats["titlecase_ratio"] = float(n_title) / words_denom

        # Readability metrics
        syllables = [estimate_syllables(w) for w in clean_words_lower] if clean_words_lower else [1]
        total_syllables = sum(syllables)
        avg_syllables = float(total_syllables) / words_denom
        feats["avg_syllables_per_word"] = avg_syllables

        # Flesch Reading Ease & Kincaid Grade Level
        # 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        flesch_ease = 206.835 - 1.015 * mean_wps - 84.6 * avg_syllables
        feats["flesch_reading_ease"] = float(np.clip(flesch_ease, -50.0, 150.0))
        # 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        kincaid_grade = 0.39 * mean_wps + 11.8 * avg_syllables - 15.59
        feats["kincaid_grade_level"] = float(np.clip(kincaid_grade, 0.0, 30.0))

        # 4. Discourse Markers & Lab Fingerprints (Normalized per 1000 words)
        words_per_1000 = words_denom / 1000.0
        for marker in DISCOURSE_MARKERS:
            # Word boundary regex search for multi-word or single-word marker
            m_cnt = len(re.findall(r"\b" + re.escape(marker) + r"\b", text_lower))
            feats[f"marker_{marker.replace(' ', '_')}"] = float(m_cnt) / words_per_1000

        return feats
