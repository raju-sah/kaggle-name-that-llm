"""Helper utility to check live Name That LLM submissions and submit v4 Dual T4 GPU kernels."""

import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionLimitsRequest


def main() -> int:
    api = KaggleApi()
    api.authenticate()

    print("=== Current Submissions (name-that-llm) ===")
    subs = api.competition_submissions("name-that-llm")
    submitted_v4_slugs = set()
    for s in subs:
        desc = getattr(s, "description", "") or ""
        pub = getattr(s, "_public_score", None)
        err = getattr(s, "error_description", None)
        print(f"Ref: {s.ref} | Date: {s.date} | Status: {s.status} | Public LB: {pub} | Desc: {desc}")
        if err:
            print(f"  -> Error: {err}")
        if ("v4 Dual T4 GPU" in desc or "v5 Dual T4 GPU" in desc or "v6 Dual T4 GPU" in desc) and "EXP-002" in desc:
            submitted_v4_slugs.add("rajucode/name-that-llm-exp002")
        if ("v4 Dual T4 GPU" in desc or "v5 Dual T4 GPU" in desc or "v6 Dual T4 GPU" in desc) and "EXP-003" in desc:
            submitted_v4_slugs.add("rajucode/name-that-llm-exp003")

    with api.build_kaggle_client() as kaggle:
        lreq = ApiGetSubmissionLimitsRequest()
        lreq.competition_name = "name-that-llm"
        limits = kaggle.competitions.competition_api_client.get_submission_limits(lreq)
        allowed_now = getattr(limits, "_num_allowed_now", 0)
        num_today = getattr(limits, "_num_today", 0)
        print(f"\n=== Daily Submission Quota: used={num_today}/5, allowed_now={allowed_now} ===")

    targets = [
        (
            "rajucode/name-that-llm-exp002",
            6,
            "EXP-002 (v6 Dual T4 GPU): Two-Stage LoRA (8B+14B) + 4-Gaussian Stack + 10% ATLAS + Curly/EmDash Invariants + Hungarian 9-18",
        ),
        (
            "rajucode/name-that-llm-exp003",
            6,
            "EXP-003 (v6 Dual T4 GPU): Apex ATLAS + Temp 0.92 + Two-Tier DP Injective Marginals (2-8) & Hungarian (9-18)",
        ),
    ]

    for slug, ver, msg in targets:
        if slug in submitted_v4_slugs:
            print(f"[SKIP] {slug} v{ver} already submitted.")
            continue
        if allowed_now <= 0:
            print(f"[WAIT] {slug} v{ver} is committed & verified on Dual T4 GPUs; waiting for daily quota reset (00:00 UTC).")
            continue
        try:
            res = api.competition_submit_code("submission.parquet", msg, "name-that-llm", slug, ver)
            print(f"[SUCCESS] Submitted {slug} v{ver}: {res}")
            allowed_now -= 1
        except Exception as e:
            print(f"[ERROR] Failed submitting {slug} v{ver}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
