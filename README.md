# kaggle-name-that-llm

Solution workspace for **Kaggle — Name That LLM** (Featured Code Competition, $50,000).

> **Identify the model behind the text.**

This repo is built around the central question circulating in the [competition discussion](https://www.kaggle.com/competitions/name-that-llm/discussion/749447):

> *"Will the winner just be whoever can get the most amount of extra training data?"*

The short answer, based on the evidence posted in that thread: **no — but only if your extra data looks like the host's data.** Raw volume is close to worthless; distribution match is everything.

---

## Competition at a glance

| | |
|---|---|
| **Competition** | [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) (id `170484`) |
| **Type** | Featured **Code** Competition — you submit a *notebook*, not a file |
| **Host** | Kaggle |
| **Prize pool** | **$50,000 USD** across 10 prizes |
| **Launched** | 2026-10-14 |
| **Final deadline** | **2027-01-14 23:59 UTC** |
| **Lockdown deadlines** | 2027-01-06 — team mergers, new entrants, and kernel publishing all close |
| **Metric** | **Balanced Accuracy** over 18 classes |
| **Public leaderboard** | 40% of test data (final standings computed on the other 60%) |
| **Max team size** | 5 |
| **Submissions** | 5/day, **2** final submissions selected |
| **Runtime limit** | 540 min (9h) on CPU **or** GPU; **internet disabled** during scoring |
| **Required output** | `submission.parquet` in `/kaggle/working` |
| **Data license** | CC BY-SA 4.0 |
| **Field size (2026-10-19)** | 1,887 competitors · 1,824 teams · 9,976 submissions |

---

## Task & data

Each passage is an authentic completion from one of 18 LLMs answering a prompt. Predict which model wrote it.

**`train.csv`** — 2,699 passages written for **150 prompts**.

| column | type | notes |
|---|---|---|
| `id` | str | anonymised passage id |
| `prompt_id` | str | anonymised prompt id |
| `text` | str | generated passage |
| `model` | str | ground truth (one of the 18) |

**`test.csv`** — `id`, `text`. ~4,000 hidden passages on **prompts not present in `train.csv`**. During interactive sessions the file holds a single example row so your notebook runs end to end; it is re-run against the full hidden set on submission.

**`sample_submission.parquet`** — `id`, `model`.

Passages are self-contained **single paragraphs** — only light formatting cleanup was applied (e.g. a whole-passage `**bold**` wrapper removed).

---

## Metric — Balanced Accuracy

Plain accuracy would let the majority models dominate. Balanced accuracy weights all 18 models equally:

$$
\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \frac{\left| i : y_i = k \land \hat{y}_i = k \right|}{\left| i : y_i = k \right|}
$$

with $K = 18$, $y_i$ the true authoring model and $\hat{y}_i$ your prediction.

**Test passages per model are not equal, and the model mix differs between public and private leaderboards.** A model that is never predicted for will score 0 on that class and drag the average down — so always predict *all 18* labels, and expect the public/private gap to be non-trivial.

---

## The 18 candidate models

| Provider | Model | | Provider | Model |
|---|---|---|---|---|
| OpenAI | `gpt-5.5` | | Anthropic | `claude-fable-5` |
| OpenAI | `gpt-5.6-luna` | | Anthropic | `claude-opus-4-6` |
| OpenAI | `gpt-5.6-sol` | | Anthropic | `claude-opus-4-8` |
| OpenAI | `gpt-5.6-terra` | | Anthropic | `claude-opus-5` |
| OpenAI | `gpt-6-astra` | | Anthropic | `claude-sonnet-5` |
| Google | `gemini-3.1-pro-preview` | | Qwen | `qwen3.8-max` |
| Google | `gemini-3.6-flash` | | ZAI | `glm-5.3` |
| Google | `gemini-3.7-flash` | | ZAI | `glm-5.3-flash` |
| Google | `gemini-3.8-flash` | | Moonshot AI | `kimi-k3` |

OpenAI 5 · Anthropic 5 · Google 4 · ZAI 2 · Qwen 1 · Moonshot AI 1.

Note the label set is mostly *same-family* variants (`gpt-5.6-luna` / `sol` / `terra`, `claude-opus-4-6` / `4-8` / `5`, four `gemini-*` flashes). Separating siblings is the hard part of this competition, and it is where provider-distinctive tokens, refusal style, list/heading habits, and length calibration pay off.

---

## Submission format

Read the test set from `/kaggle/input/competitions/name-that-llm/test.csv` and write **`submission.parquet`** (parquet, not csv) to the notebook's working directory.

```python
import pandas as pd

test = pd.read_csv('/kaggle/input/competitions/name-that-llm/test.csv')
submission = pd.DataFrame({'id': test['id'], 'model': predictions})
submission.to_parquet('submission.parquet', index=False)
```

Rejected if: missing `id`/`model` column, null values, repeated `id`, or a label outside the 18 above.

---

## The debate: is this "pay to win"?

Source thread: [Will the winner just be whoever can get the most amount of extra training data?](https://www.kaggle.com/competitions/name-that-llm/discussion/749447) — opened by [CoreyJamesLevinson](https://www.kaggle.com/returnofsputnik) on launch day, 92 upvotes and 22 comments, with more joining since.

### The question

> *"Someone who has 1M training examples per model will find more patterns than someone with 100K training examples per model. Is paying model providers for data generation going to be the key to winning?"*

### The empirical answer

The most useful data point in the thread came from [HALX2158](https://www.kaggle.com/mizuno0522) (147th place), who ran a controlled experiment: char+word TF-IDF + LinearSVC, GroupKFold **by prompt**, scored on competition texts only.

| Training data | Balanced Accuracy |
|---|---|
| Competition data only | **0.326** |
| \+ 850 external rows (already single paragraphs) | **0.308** |
| \+ 14,911 external rows (flattened to one paragraph) | **0.279** |

The same direction held with a stronger TF-IDF and with **frozen Qwen3-8B embeddings**. More external data made it *worse*.

The suspected cause is a **distribution mismatch**, not a data-volume problem:

| Property | Competition texts | External answers |
|---|---|---|
| Line breaks | 0% | 94% |
| Bold | 0% | 60% |
| Median length | 101 words | 174 words |

> *"So 'more data' may matter, but only if it looks like the host's data."*

Details and code: [model-fingerprints-punctuation-external-data](https://www.kaggle.com/code/mizuno0522/model-fingerprints-punctuation-external-data)

### The positions

**"Resources are a prerequisite, not a cheat."** — [Yurnero](https://www.kaggle.com/samson8) (8 golds) argues featured comps are meant to be competed at the top of the field; of his 8 golds only 2 needed ≥$1000 of compute, the other 6 were winnable on free Kaggle GPUs. He also points at the *positive* framing: a model that can separate same-family LLMs is a form of **audit** — proof that providers shipped the models you paid for. He'd be happy to see such a solution even if he couldn't afford to build it. He also notes the practical difficulty: you can't prove a team used the data they claim unless you re-run their pipeline end to end and land near their weights.

**"Pay-to-play is still pay-to-play."** — [Cody_Null](https://www.kaggle.com/cody11null) (4 golds) agrees Kaggle is owed a return and that hosts should set the resource envelope, but argues that a competition where ~$5k of extra dataset creation buys a decisive edge crosses a line, especially for students and competitors from less GPU-rich countries. He points to the Kaggle ruleset requiring solution resources to stay at a reasonable level to preserve fairness. His practical suggestion for anyone trying this: compile your own corpus, use the competition train set as **validation**, and if it stops correlating, treat your data as additive only.

**"Ban external data or this competition."** — [DeepUnderstanding](https://www.kaggle.com/mrinath) (17 upvotes) — the maximalist position.

**"Even with a huge dataset, the barrier just moves."** — [greySnow](https://www.kaggle.com/shlomoron): *"It's rare to see so blatant 'pay to win' competition as this one 🤣🤣 Although even if we got a large dataset participant can still generate more for an advantage…and then if we have huge amount of data the barrier would just move to pay to win for compute."*

**"I don't even understand the point of this competition."** — [Zejun_](https://www.kaggle.com/zejunfool): *"after all, aren't these companies just distilling each other's models?"* [Noa Roggendorff](https://www.kaggle.com/nroggendorff) pushes back on the premise: architecture and post-training genuinely differ, the RL pipeline is where behavioural traits come from — and reports the practical difficulty that *"I'm having a hard time telling the difference between the ZAI models."*

**"The hosts should just provide the data."** — [Bharat](https://www.kaggle.com/bharat0): *"the winners might have to spend more than they win (and pay taxes)."* (see the attached host-dashboard screenshots in the thread). [Ayush Khaire](https://www.kaggle.com/ayushkhaire) raises the matching technical risk: with ~4,000 train and ~4,000 test passages, if you train on external data generated by a different pipeline than the host's, the prompt↔response patterns may simply not transfer.

**"Who's the model-router company going to win?"** — [hsiaosuan](https://www.kaggle.com/hsiaosuan): whoever owns a routing layer gets the fingerprints for free.

### What is still unresolved

> **Open question — ask the host.** [wsc-MISX](https://www.kaggle.com/wscmisx) asked the organisers directly: *are participants allowed to generate additional training data by calling LLM APIs themselves?* The training set has only ~150 unique prompts, which is clearly limited. Specifically: **how exactly will the "Reasonableness Standard" in Section 8.2 be defined and applied?** Is there a specific cost threshold for API usage, and would paid-API-generated training data count as acceptable? As of 2026-10-17 no host reply had appeared.

[Ayush Khaire](https://www.kaggle.com/ayushkhaire) adds a hypothesis worth watching: if the public and private leaderboards were derived from a single dataset, external data may help; if the host generated the test set differently, it may not. [Tawara](https://www.kaggle.com/ttahara) frames the same thing as a **domain gap**: *"the simplest way to eliminate this gap is to obtain the training data directly via LLMs' API."*

---

## Rules that actually bite

**External data and pre-trained models** may be used **only if** they are freely and publicly available and equally accessible at no cost, **or** they meet the rules' **Reasonableness Standard** — which accepts a small subscription (the rules cite something like Gemini Advanced) but *not* a proprietary dataset costing more than the prize. The rules do **not** explicitly address API-generated training data; treat it as an open question until the host answers, and never price your solution above the prize.

**Internet is off** during scoring. Models and packages must be attached as Kaggle inputs or installed via the Dependency Manager — you cannot call a hosted model at scoring time. 4 CPU cores / 30 GB RAM, or 1× P100 or 2× T4 with 29 GB RAM; free GPU quota ~30 h/week.

**Also binding:** no hand-labelling or human prediction of validation/test records; no private code sharing outside your team; any open-source code must carry an OSI-approved licence that does not limit commercial use; competition data is CC BY-SA 4.0 but must not be shared with anyone who has not accepted the rules.

**Note on the sponsor** — a community claim in the thread (not confirmed by the host) that Google sponsors the competition officially, with OpenAI and Anthropic as "shadow sponsors", and Qwen / GLM / Kiwi added to broaden the model pool.

---

## Strategy implications

1. **Match the host's distribution before you add volume.** Single paragraph, no line breaks, no bold, ~101-word median. Normalise external data to look like the host's — the 14,911-row corpus scored *worse* than no external data at all, and a simple flatten to one paragraph is the difference between 0.279 and 0.308.
2. **Group your validation by prompt, not by row.** Test prompts are disjoint from train prompts. Random-split CV leaks; `GroupKFold` on `prompt_id` is the honest estimate.
3. **Use the competition train set as validation for any external corpus.** If your external data stops correlating, downweight or drop it.
4. **Target the same-family pairs.** Five `gpt-5.6-*`, four `gemini-*` flashes, three `claude-opus-*`. Balanced accuracy means each sibling class is worth as much as a whole provider.
5. **Never predict zero for a class.** Unpredicted classes score 0 recall and are averaged in at equal weight.
6. **Have a zero-cost path.** The fairness argument is loud enough that a solution resting on paid API generation is both a rules risk and a legitimacy risk. A competition-only solution that works is worth more than a marginally better one that depends on a questionable spend.
7. **Use the full submission budget.** 5 submissions/day, 2 final selections, deadline 2027-01-14, kernel publishing closes 2027-01-06. Public LB is only 40% of the test set and the class mix differs on private — validate against that gap, not against the public number.

---

## Repo layout

```
kaggle-name-that-llm/
├── README.md          # this file
├── notebooks/         # Kaggle notebooks (submission-ready)
├── src/               # local training / feature pipeline
├── data/              # local copies of train.csv / sample_submission
└── notes/             # experiment log, leaderboard snapshots, writeups
```

---

## Links

- [Competition overview](https://www.kaggle.com/competitions/name-that-llm/overview)
- [Data](https://www.kaggle.com/competitions/name-that-llm/data)
- [Rules](https://www.kaggle.com/competitions/name-that-llm/rules)
- [Leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard)
- [Discussion](https://www.kaggle.com/competitions/name-that-llm/discussion) · [this thread](https://www.kaggle.com/competitions/name-that-llm/discussion/749447)
- [HALX2158's external-data experiment](https://www.kaggle.com/code/mizuno0522/model-fingerprints-punctuation-external-data)
- [Kaggle Benchmarks](https://www.kaggle.com/benchmarks) — $100/month token quota per API
