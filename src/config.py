from pathlib import Path
from typing import Dict, List

# Root repository directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_RAW_DIR / "sample_submission.parquet"
FOLDS_PATH = DATA_PROCESSED_DIR / "cv_folds.parquet"

# Artifacts paths
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
OOF_DIR = ARTIFACTS_DIR / "oof"
MODELS_DIR = ARTIFACTS_DIR / "models"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"

# Notes & recipes paths
NOTES_DIR = ROOT_DIR / "notes"
COOKBOOKS_DIR = ROOT_DIR / "cookbooks"
EXPERIMENTS_LOG_PATH = NOTES_DIR / "experiments.jsonl"
PROGRESS_PATH = NOTES_DIR / "progress.md"
DISCOVERIES_PATH = NOTES_DIR / "discoveries.md"
CLOSED_IDEAS_PATH = NOTES_DIR / "closed_ideas.md"
PLAN_PATH = NOTES_DIR / "plan.md"

# Canonical sorted list of 18 models
LABELS: List[str] = [
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

NUM_CLASSES: int = len(LABELS)

LABEL_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}

# Model provider taxonomy
PROVIDER_MAP: Dict[str, str] = {
    "claude-fable-5": "anthropic",
    "claude-opus-4-6": "anthropic",
    "claude-opus-4-8": "anthropic",
    "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    "gemini-3.1-pro-preview": "google",
    "gemini-3.6-flash": "google",
    "gemini-3.7-flash": "google",
    "gemini-3.8-flash": "google",
    "glm-5.3": "zai",
    "glm-5.3-flash": "zai",
    "gpt-5.5": "openai",
    "gpt-5.6-luna": "openai",
    "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai",
    "gpt-6-astra": "openai",
    "kimi-k3": "moonshot",
    "qwen3.8-max": "qwen",
}

# Provider to list of models
FAMILY_MODELS: Dict[str, List[str]] = {}
for model, provider in PROVIDER_MAP.items():
    FAMILY_MODELS.setdefault(provider, []).append(model)

# Validation settings
N_FOLDS: int = 5
RANDOM_SEED: int = 42
