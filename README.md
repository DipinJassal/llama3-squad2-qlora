# LLaMA-3-8B on SQuAD 2.0: Hallucination vs. Abstention

Does LLaMA-3-8B-Instruct make up an answer when a question can't be answered from the passage,
or does it correctly say "no answer"? This project measures that on
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), where half of the dev-set questions are
deliberately unanswerable. It then tries to improve the model with prompting and QLoRA fine-tuning.

> **Status: work in progress.** Data exploration and the evaluation metrics are done.
> Model results will be added only from real runs. No numbers here are estimated.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 0 | Setup and data exploration | ✅ Done |
| 1 | SQuAD 2.0 metrics + hallucination / over-abstention rates | ✅ Done, verified against the official scorer |
| 2 | Zero-shot baseline with LLaMA-3-8B-Instruct on vLLM | ⏳ Next |
| 3 | LangGraph pipeline: load → prompt → generate → parse → [judge] → score | Planned |
| 4 | Prompt variants: zero-shot, abstention instruction, few-shot with unanswerable examples, self-check | Planned |
| 5 | LLM-as-a-judge, validated against human labels (agreement + Cohen's kappa) | Planned |
| 6 | QLoRA fine-tuning of LLaMA-3-8B | Planned |
| 7 | Ablation: LoRA target modules (q,v / q,k,v,o / attention+MLP) × rank (8, 16, 32, 64) | Planned |
| 8 | Results tables, params-vs-EM plot | Planned |

## The data

| Split | Questions | Answerable | Unanswerable |
|---|---|---|---|
| train | 130,319 | 86,821 | 43,498 (33.4%) |
| dev (`validation`) | 11,873 | 5,928 | 5,945 (50.1%) |

Unanswerable questions have an empty answer list. They were written by crowdworkers to *look*
answerable. They reuse the passage's wording and change one detail. Example passage: "Rollo ...
agreed to swear fealty to King Charles III". Unanswerable question: *"Who did King Charles III
swear fealty to?"* The tempting wrong answer, "Rollo", appears right there in the text.

Because dev is 50/50, a model that **always** abstains scores 50.1% overall EM while answering
nothing. Overall EM alone is therefore misleading, so this project reports the metrics below.

## Metrics

Implemented in [`metrics.py`](metrics.py). A prediction of `""` means the model abstained.

| Metric | Computed over | Meaning |
|---|---|---|
| EM / F1 | all questions | Exact match / token-overlap F1 after SQuAD normalization, max over reference answers |
| HasAns EM / F1 | answerable questions | How well it answers when there *is* an answer |
| NoAns accuracy | unanswerable questions | How often it correctly abstains |
| Hallucination rate | unanswerable questions | % where it gave an answer anyway (= 100 − NoAns accuracy) |
| Over-abstention rate | answerable questions | % where it said "no answer" although one exists |

Hallucination rate and over-abstention rate are reported together because each can be
trivially gamed on its own: never answering gives 0% hallucination, and always answering gives
0% over-abstention.

**Verification.** [`check_metrics.py`](check_metrics.py) scores the same predictions with this
code and with Hugging Face's official `squad_v2` metric. The two agree exactly on a hand-checked
toy set and on synthetic predictions for all 11,873 dev questions. The cross-check caught a real
edge case: three answerable dev questions include a junk reference answer of `"."`, which the
official scorer discards. Our first version didn't, which would have wrongly rewarded abstaining.

## Repository layout

| File | Purpose |
|---|---|
| `explore_data.py` | Loads SQuAD 2.0, counts answerable vs unanswerable, shows a passage with its questions |
| `metrics.py` | Normalization, EM, F1, aggregate metrics; run it to execute the built-in tests |
| `check_metrics.py` | Cross-check against the official `squad_v2` scorer |
| `LEARNING_LOG.md` | Per-stage notes: what was built, decisions, and the real numbers obtained |

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python explore_data.py     # dataset statistics
python metrics.py          # metric unit tests
python check_metrics.py    # compare against the official scorer
```

Stages 0–1 run on a CPU. From Stage 2 on, an NVIDIA GPU is required (vLLM and bitsandbytes need
CUDA), as is approved access to the gated
[`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
model (`hf auth login`).
