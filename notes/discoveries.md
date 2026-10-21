# Scientific Discoveries & Causal Insights

> Distilled scientific truths about the competition dataset, model behaviors, and performance drivers.
> Must be kept concise (< 250 lines) to prevent context rot. Verbose analysis belongs in `notes/archive/`.

---

### D-001: Data Geometry & Prompt Pairing
- `train.csv` contains exactly 2,699 passages across 150 unique `prompt_id`s.
- 149 prompts have completions from all 18 models; 1 prompt has 17 completions.
- Because nearly every prompt is answered by every model, the dataset is structured as a nearly complete $150 \times 18$ grid.
- This grid structure enables:
  1. Supervised Contrastive Learning (SupCon) using same-prompt passages as hard negatives.
  2. Within-prompt feature demeaning to isolate authorship signature from topic noise.

### D-002: Class Taxonomy Breakdown
- 18 classes are divided into 4 multi-model families and 2 singletons:
  - **OpenAI (5):** `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
  - **Anthropic (5):** `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`
  - **Google (4):** `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
  - **ZAI (2):** `glm-5.3`, `glm-5.3-flash`
  - **Singletons (2):** `kimi-k3` (Moonshot), `qwen3.8-max` (Qwen)
- Separating providers is relatively straightforward (e.g. Qwen vs OpenAI); the vast majority of classification errors occur between same-family siblings.

### D-003: EXP-001 Baseline & Sibling Bottleneck
- Multi-granular TF-IDF (word 1-2, char 3-5) + 165 stylometric features with a Ridge/SGD ensemble achieves **0.6091 Provider Accuracy** on 5-fold CV, but only **0.2849 Balanced Accuracy** across all 18 classes (Dirichlet p05: 0.2582).
- The worst-performing classes are sibling variants within families: `glm-5.3-flash` (0.1333 recall vs `glm-5.3`), `gpt-5.6-terra` (0.1600 recall vs other GPT-5.6 variants), and `claude-opus-4-8` (0.1733 recall vs Opus 4-6 and Opus 5).
- This confirms that provider-level identification is solid, but fine-grained sister model disambiguation is the core bottleneck that requires non-linear GBDT or deep contrastive embeddings (EXP-002 / EXP-003).

### D-004: Forum Intelligence & Test-Time Balancing
- Punctuation forensics (em-dash rate, comma density, type-token ratio, sentence length standard deviation) are top-ranked features, but hard-coded punctuation rules risk brittle failure on new prompts.
- Community simulations (topic #750213) revealed that naive prompt-centering on small test clusters removes author signal alongside topic noise.
- Sinkhorn marginal balancing / class multiplier calibration prevents training-prior bias and protects against zero-recall penalties on unseen test splits without assuming an equal test distribution.

### D-005: First Live Kaggle Submission (EXP-001) & Metric Alignment
- Submission ref `57472877` executed successfully on Kaggle's hidden test set (132.8s runtime) and verified full end-to-end format compliance (`submission.parquet`, 18 classes).
- Public Leaderboard score: **0.269** (Rank #2229 / 2286).
- Score tracks 5-fold CV (**0.2849**) within -0.0159 delta, well inside the predicted Dirichlet prior-shift 5th-percentile confidence bound (0.2582).
- Causal gap to community top linear baseline (Adisak Sukul, LB 0.353 in #750390):
  1. `lowercase=False` is critical: keeping original casing preserves model-specific capitalization habits.
  2. Feature scale: Sukul used 300k char n-grams (1-5) and 200k word n-grams (1-2) with regex punctuation tokens `(?u)\b\w+\b|[^\w\s]`, totaling 500k features (vs our 25k).
  3. LogisticRegression with C=10 vs conservative Ridge/SGD.

### D-006: Early Transformer Layers Capture Authorship vs Late Layers Capturing Topic
- Forum audit of topic #750462 (Aleksey Osipenko) revealed that earlier transformer layers (e.g. Layer 3 of frozen Qwen3-4B) achieve **0.391 Balanced Accuracy** vs Layer 36 (last layer) at **0.311**.
- Topic noise is concentrated in late layers, whereas author style lives in early/intermediate layers (word choice, punctuation, rhythm).
- Blending early Qwen3 layer embeddings with TF-IDF + stylometry achieved **0.439 CV / 0.420 Public LB**.

### D-007: Fast-Commit Pre-Flight Verification & GPU Ensembling Protocol
- Heavy transformer inference (LoRA 8B/14B, 4-Gaussian stack, ATLAS) on the 2,699 training passages during Kaggle interactive commit on 3 dummy test rows takes >3.4 hours (12,336s), causing timeout risk and blocking rapid submission iterations.
- Guarding the GPU forward passes with `if len(test) >= 50:` while executing a rigorous pre-flight asset verification on `len(test) < 50` (locating LoRA checkpoints, verifying base model configs, checking ATLAS sha256 digests, and compiling source strings) allows commit runs to complete cleanly in ~240–360s.
- During competition scoring, Kaggle runs the notebook on the hidden test set (`len(test) >= 50`), activating the full two-stage LoRA + Gaussian stack + ATLAS ensemble and injective post-processing.

### D-008: Hidden Rerun Defensive Fallback Architecture & Non-Fatal Model Degradation
- Kaggle code competition reruns execute in an isolated container without internet. If any kernel data source (such as `lightningv08/ntl-lora24-fivefold-training`) becomes inaccessible (HTTP 403 Forbidden or deleted), Kaggle strips the input mount without warning.
- Unhandled `assert hits` or unhandled exceptions in any individual neural pass (LoRA, Gaussian stack, ATLAS) cause an immediate fatal rerun failure: `Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset`.
- Every neural backbone pass is wrapped in defensive try/except handling and model availability checks, with graceful fallback to the 540,000-feature 4-view TF-IDF LinearSVC logit blend. Dual output paths (`/kaggle/working/submission.parquet` and `./submission.parquet`) guarantee an intact submission regardless of working directory shifts.

### D-009: Dual Tesla T4 (`NvidiaTeslaT4`) Machine Shape, Session Cancellation Forensics & LoRA24 Weight Compensation
- **Dual Tesla T4 Requirement:** `Qwen3-14B` (`lora14b_v1`) requires ~28 GB VRAM in fp16 and splits layers across 2 GPUs (`if gib > 13 and torch.cuda.device_count() > 1: max_memory={0: "13GiB", 1: "13GiB"}`). Kernel metadata must explicitly specify `"enable_gpu": true`, `"machine_shape": "NvidiaTeslaT4"`, and `"docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"` to provision `['Tesla T4', 'Tesla T4']` (`GPUs=2`).
- **Session ID Extraction & Cancellation:** Pushing a new kernel version (`v2`/`v3`) does not terminate an already-running `v1` commit session. The active `kernel_session_id` for any version `v` can be extracted from the `/kf/<kernel_session_id>/` signed URL raised by `ApiDownloadKernelOutputRequest(version_number=v)` and terminated via `ApiCancelKernelSessionRequest(kernel_session_id=sid)`, immediately freeing `timeReserved` and batch GPU slots.
- **LoRA24 Weight Compensation:** When `lora24_public` (`lightningv08/ntl-lora24-fivefold-training`) is unmounted due to privacy changes, transferring its `0.177` plain weight and `0.15` centred-twin weight to the identical 24-layer Qwen3-8B checkpoint `lora24_seed7` (`lucasboesen/name-that-llm-train-lora24-s7`, `W=0.290`, `LORA_TWIN_W=0.15`) preserves 100% of the 24-layer Qwen3-8B LoRA ensemble signal.


