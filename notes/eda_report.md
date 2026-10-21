# Forensic EDA Report: Author Style Fingerprints

## 1. Dataset Geometry & Matrix Balance
- **Total passages:** 2699
- **Unique prompts:** 150
- **Unique models:** 18 (expected 18)
- **Prompts with full 18 models:** 149 / 150
- **Incomplete prompts:** 1 prompts have fewer than 18 models:
  - `p002`: 17 completions (missing: {'claude-fable-5'})

## 2. Model Length, Syntax & Punctuation Fingerprints

| Provider | Model | Samples | Mean Words (±Std) | Mean Sents | Words/Sent | Em-dash Rate | En-dash Rate | Semicolon Rate | Colon Rate | Markdown % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Anthropic** | `claude-fable-5` | 149 | 112.5 (±58.2) | 4.7 | 24.1 | 83.2% | 0.0% | 33.6% | 59.1% | 0.0% |
| **Anthropic** | `claude-opus-4-6` | 150 | 115.1 (±60.9) | 4.3 | 27.2 | 75.3% | 0.0% | 15.3% | 42.0% | 3.3% |
| **Anthropic** | `claude-opus-4-8` | 150 | 123.6 (±61.6) | 5.1 | 23.9 | 62.7% | 0.0% | 24.0% | 49.3% | 2.7% |
| **Anthropic** | `claude-opus-5` | 150 | 123.4 (±62.3) | 5.2 | 23.8 | 81.3% | 1.3% | 54.0% | 69.3% | 0.7% |
| **Anthropic** | `claude-sonnet-5` | 150 | 120.0 (±64.3) | 4.6 | 26.2 | 73.3% | 0.0% | 23.3% | 50.7% | 1.3% |
| **Google** | `gemini-3.1-pro-preview` | 150 | 119.3 (±59.6) | 5.8 | 20.3 | 0.0% | 0.0% | 19.3% | 16.7% | 0.0% |
| **Google** | `gemini-3.6-flash` | 150 | 111.2 (±54.1) | 4.6 | 24.7 | 0.0% | 0.0% | 22.7% | 20.0% | 0.0% |
| **Google** | `gemini-3.7-flash` | 150 | 117.1 (±59.8) | 4.8 | 23.9 | 0.7% | 0.0% | 23.3% | 22.7% | 1.3% |
| **Google** | `gemini-3.8-flash` | 150 | 100.7 (±52.8) | 4.4 | 22.7 | 0.0% | 0.0% | 28.0% | 18.0% | 0.0% |
| **Moonshot** | `kimi-k3` | 150 | 107.4 (±53.7) | 4.5 | 25.7 | 52.0% | 1.3% | 61.3% | 77.3% | 0.0% |
| **Openai** | `gpt-5.5` | 150 | 110.3 (±54.7) | 4.3 | 26.3 | 13.3% | 0.0% | 60.7% | 65.3% | 0.0% |
| **Openai** | `gpt-5.6-luna` | 150 | 115.4 (±57.8) | 4.9 | 23.7 | 36.7% | 1.3% | 42.7% | 70.7% | 0.7% |
| **Openai** | `gpt-5.6-sol` | 150 | 118.9 (±60.5) | 4.9 | 24.9 | 27.3% | 1.3% | 62.0% | 64.0% | 0.0% |
| **Openai** | `gpt-5.6-terra` | 150 | 113.3 (±54.5) | 4.4 | 26.3 | 22.7% | 1.3% | 52.0% | 73.3% | 0.0% |
| **Openai** | `gpt-6-astra` | 150 | 117.8 (±59.1) | 5.5 | 21.1 | 8.7% | 0.0% | 52.7% | 65.3% | 0.0% |
| **Qwen** | `qwen3.8-max` | 150 | 108.3 (±56.8) | 5.0 | 22.2 | 3.3% | 0.0% | 39.3% | 47.3% | 0.0% |
| **Zai** | `glm-5.3` | 150 | 110.3 (±56.8) | 4.9 | 23.3 | 82.0% | 0.0% | 34.0% | 62.0% | 0.0% |
| **Zai** | `glm-5.3-flash` | 150 | 119.2 (±58.2) | 5.4 | 23.2 | 79.3% | 0.7% | 48.0% | 65.3% | 0.0% |

## 3. Provider Family Level Contrasts

| Provider | Avg Word Count | Avg Sentence Count | Em-dash Usage Rate | Semicolon Rate | Colon Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Anthropic** | 119.0 | 4.8 | 75.2% | 30.0% | 54.1% |
| **Google** | 112.1 | 4.9 | 0.2% | 23.3% | 19.3% |
| **Moonshot** | 107.4 | 4.5 | 52.0% | 61.3% | 77.3% |
| **Openai** | 115.1 | 4.8 | 21.7% | 54.0% | 67.7% |
| **Qwen** | 108.3 | 5.0 | 3.3% | 39.3% | 47.3% |
| **Zai** | 114.8 | 5.1 | 80.7% | 41.0% | 63.7% |

## 4. Key Forensic Discoveries for Modeling
1. **Zero Newlines & Clean Formatting:** 100% of competition passages contain 0 newlines (`\n`), confirming single self-contained paragraphs.
2. **Punctuation Fingerprints are Discriminative:** Em-dash (`—`), en-dash (`–`), semicolons, and colons exhibit stark differences across providers and siblings.
3. **Sibling Length Calibrations:** Significant length variations exist even within families (e.g. flash vs pro models, opus vs sonnet).
4. **Discourse Rhythm:** Sentence length variance and burstiness strongly separate human-like vs concise instruction-tuned models.
