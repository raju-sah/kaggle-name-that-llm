import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from src.config import (
    CLOSED_IDEAS_PATH,
    DISCOVERIES_PATH,
    EXPERIMENTS_LOG_PATH,
    LABELS,
    NOTES_DIR,
    NUM_CLASSES,
    OOF_DIR,
    PLAN_PATH,
    PROGRESS_PATH,
    ROOT_DIR,
    TRAIN_PATH,
)


def run_baton_check() -> bool:
    """Verifies that all relay baton artifacts and OOF registry files are consistent."""
    errors: List[str] = []
    warnings: List[str] = []

    print("Running Relay Baton Health Check...")

    # 1. Check AGENTS.md
    agents_path = ROOT_DIR / "AGENTS.md"
    if not agents_path.exists():
        errors.append("AGENTS.md is missing from repository root!")
    else:
        content = agents_path.read_text()
        if "Balanced Accuracy" not in content or "Pass-The-Baton" not in content:
            warnings.append("AGENTS.md seems incomplete or missing key protocol sections.")

    # 2. Check notes/plan.md has [NEXT]
    if not PLAN_PATH.exists():
        errors.append("notes/plan.md is missing!")
    else:
        plan_content = PLAN_PATH.read_text()
        if "[NEXT]" not in plan_content:
            errors.append("notes/plan.md does not contain any '[NEXT]' task pointer for the next agent!")

    # 3. Check progress.md and experiments.jsonl
    if not PROGRESS_PATH.exists():
        errors.append("notes/progress.md is missing!")

    logged_exp_ids = set()
    if EXPERIMENTS_LOG_PATH.exists() and EXPERIMENTS_LOG_PATH.stat().st_size > 0:
        with open(EXPERIMENTS_LOG_PATH, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "exp_id" in record:
                        logged_exp_ids.add(record["exp_id"])
                except Exception as e:
                    errors.append(f"experiments.jsonl line {line_no} is invalid JSON: {e}")

    # 4. Check OOF Parquet files
    if TRAIN_PATH.exists():
        n_train = len(pd.read_csv(TRAIN_PATH))
    else:
        n_train = 2699

    oof_files = list(OOF_DIR.glob("*.parquet")) if OOF_DIR.exists() else []
    print(f"Found {len(oof_files)} OOF prediction files in {OOF_DIR.relative_to(ROOT_DIR)}.")

    expected_cols = {"id", "fold", "pred", "true"}.union(set(LABELS))

    progress_text = PROGRESS_PATH.read_text() if PROGRESS_PATH.exists() else ""

    for oof_file in oof_files:
        try:
            df_oof = pd.read_parquet(oof_file)
            missing_cols = expected_cols - set(df_oof.columns)
            if missing_cols:
                errors.append(f"{oof_file.name} is missing mandatory columns: {missing_cols}")

            if len(df_oof) != n_train:
                errors.append(f"{oof_file.name} row count ({len(df_oof)}) != train rows ({n_train})")

            # Check if mentioned in progress.md or logged_exp_ids
            exp_id_candidate = oof_file.stem.split("_")[0]
            if oof_file.name not in progress_text and exp_id_candidate not in logged_exp_ids:
                warnings.append(f"{oof_file.name} is not documented in notes/progress.md or experiments.jsonl!")
        except Exception as e:
            errors.append(f"Failed to read OOF file {oof_file.name}: {e}")

    # 5. Check discoveries and closed_ideas
    if not DISCOVERIES_PATH.exists():
        errors.append("notes/discoveries.md is missing!")
    if not CLOSED_IDEAS_PATH.exists():
        errors.append("notes/closed_ideas.md is missing!")

    # Report results
    print("-" * 60)
    if warnings:
        print("[WARNINGS]:")
        for w in warnings:
            print(f"  * {w}")

    if errors:
        print("[FAILED] Baton check encountered errors:")
        for e in errors:
            print(f"  ! {e}")
        return False

    print("[SUCCESS] All Relay Baton checks PASSED! Institutional memory is clean.")
    return True


if __name__ == "__main__":
    passed = run_baton_check()
    sys.exit(0 if passed else 1)
