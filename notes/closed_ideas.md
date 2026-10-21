# Closed Ideas & Dead Ends (The Graveyard)

> **"Agents forget. Notes remember!"**
> Never repeat an experiment that failed without fundamentally changing the underlying mechanics.
> Every entry here must document: What was tried, the observed CV delta, why it failed (root cause), and what would have to change before trying again.

---

### G-001: Un-grouped Standard KFold or StratifiedKFold
- **What was tried:** Standard random row splitting across 5 folds.
- **Observed Result:** Spurious, artificially inflated CV score (up to ~0.75+), completely failing to generalize to unseen test prompts.
- **Root Cause:** 70%+ of textual variance is driven by prompt topic rather than LLM authorship style. Random splitting leaks prompt topics between train and validation.
- **Rule:** All validation must strictly use prompt-grouped splits (`data/processed/cv_folds.parquet`).

---

### G-002: Un-normalized Multi-Paragraph External Data
- **What was tried:** Appending 14,911 multi-paragraph raw external LLM answers to the training set.
- **Observed Result:** Balanced Accuracy dropped from **0.326** down to **0.279** (README benchmark from Mizuno0522).
- **Root Cause:** Massive distribution mismatch. Host competition texts have 0% line breaks, 0% outer bold wrappers, and a median length of 101 words. Raw external outputs have 94% line breaks, 60% bold, and 174-word median length.
- **Rule:** No external data may be added without passing a 2-stage normalizer (strip outer bold, enforce single paragraph, length between 35–260 words) and an adversarial domain classifier gate.

---

### G-003: Uncalibrated Posterior Argmax with Class Starvation
- **What was tried:** Taking raw `argmax_k P(y=k|x)` without class prior balancing or minimum-prediction thresholding.
- **Observed Result:** Ambiguous sibling classes (e.g., `claude-opus-4-6` vs `4-8`, or `gpt-5.6-luna` vs `terra`) received 0 predictions.
- **Root Cause:** Balanced Accuracy weights each of the 18 classes equally ($\frac{1}{18} \approx 0.0556$). When a class gets 0 predictions, its recall is 0.0, instantly costing ~0.0556 in LB penalty.
- **Rule:** All post-processors must enforce a minimum recall floor / multiplier calibration to guarantee no class is starved.

---

### G-004: End-to-End Deep Fine-Tuning During Kaggle Submission Rerun
- **What was tried:** Running 5 folds of transformer fine-tuning from scratch inside the Kaggle submission notebook.
- **Observed Result:** Notebook timeout (>540 min) or OOM on Kaggle 1x P100 / 2x T4 GPU.
- **Root Cause:** Fine-tuning DeBERTa across 5 folds with contrastive sampling takes significant GPU time; doing it at inference time is fragile and prevents model ensembling.
- **Rule:** Heavy training must run on Colab Pro CLI GPU; weights are uploaded as Kaggle Datasets (`rajucode/name-that-llm-models`) so Kaggle submission notebook only runs inference (<15 min).

---

### G-005: Test-Time Prompt Clustering & Hungarian Assignment
- **What was tried:** Clustering test passages into pseudo-prompts and enforcing 1-model-per-cluster via Hungarian assignment (`scipy.optimize.linear_sum_assignment`) and per-cluster log-prob centering (Topic #750390).
- **Observed Result:** Grouped CV promised +0.10 gain (assuming 18 answers per prompt), but on Public LB score dropped from 0.353 down to 0.329 (-0.024 drop).
- **Root Cause:** The hidden test set does not have 18 completions per prompt; prompt-level centering on small or incomplete test clusters removes author style alongside topic noise.
- **Rule:** Do not enforce uniform 18-model Hungarian assignment across test passages. Rely on independent sample inference with marginal prior-shift calibration.

---

### G-006: Unguarded Heavy Training-Pass Execution During Kaggle Notebook Commit
- **What was tried:** Executing all 2,699 training passages through 8B/14B decoder trunks, 4-Gaussian stack, and ATLAS during interactive notebook commit on 3 dummy test rows (`len(test) == 3`).
- **Observed Result:** Interactive commit runs took 12,336s (3.4 hours), exhausted batch GPU slots, and blocked subsequent version pushes with "Maximum batch GPU session count reached".
- **Root Cause:** Running full training set passes inside the notebook commit rather than guarding them behind `if len(test) >= 50:`.
- **Rule:** Guard all heavy GPU passes behind `if len(test) >= 50:`. For `len(test) < 50`, execute fast pre-flight verification on asset paths, weights, and code compilation.

---

### G-007: Unverified Private Kernel Sources & Unguarded Assertions in Scoring Rerun
- **What was tried:** Including external Kaggle kernel sources (e.g. `lightningv08/ntl-lora24-fivefold-training`) in `kernel-metadata.json` and asserting their file presence with hard assertions (`assert hits, 'error'`) inside the inference pipeline without try/except defensive guards.
- **Observed Result:** Submissions 57474896 and 57474899 crashed during hidden test rerun with `Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset`, scoring 0.000 / None.
- **Root Cause:** Kaggle silently strips forbidden/deleted/private kernel mounts when the scoring container starts. Hard assertions immediately crashed the process.
- **Rule:** Only depend on public datasets (`rajucode/name-that-llm-models`) or public notebooks with confirmed read access. Every single neural pass must have a fallback path to the 540k-feature TF-IDF linear blend so an unmounted model degrades performance rather than aborting the rerun.

---

### G-008: Submitting Multi-LLM Ensembles with `"enable_gpu": false` to Bypass Batch GPU Limits
- **What was tried:** Setting `"enable_gpu": false` in `kernel-metadata.json` (Version 3, refs `57476183` and `57476194`) when Kaggle rejected a GPU push with `Maximum batch GPU session count of 2 reached` due to older commit runs (`v1`) still executing in the background.
- **Observed Result:** Because Kaggle's hidden test rerun inherits the committed kernel version's hardware settings, a kernel committed with `"enable_gpu": false` executes the hidden test rerun on a CPU-only container (`DEV = 'cpu'`), skipping CUDA-dependent passes (`lora8b_v1`, `lora14b_v1`, `ATLAS`) and running `_GAUSSIAN_SOURCE` frozen decoders in `float32` on CPU.
- **Root Cause:** Pushing a new kernel version does not cancel older running versions of the same kernel; disabling GPU works around the commit queue but strips GPU acceleration from the competition scoring rerun.
- **Rule:** Never set `"enable_gpu": false` on GPU ensemble kernels. Extract the stale `kernel_session_id` via `ApiDownloadKernelOutputRequest` (`/kf/<session_id>/`), cancel it via `ApiCancelKernelSessionRequest`, and commit with `"enable_gpu": true` and `"machine_shape": "NvidiaTeslaT4"` (verified in Version 4).


