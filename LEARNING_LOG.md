# Learning Log

## Stage 0: Setup & data (2026-10-17)

**What we built**
- `.venv/`: Python 3.11 virtual environment (via `uv`) with `datasets` and `pandas`.
- `explore_data.py`: loads SQuAD 2.0, counts answerable vs unanswerable questions per split,
  and prints one passage with all of its questions.

**Real numbers**
| Split | Total | Answerable | Unanswerable |
|---|---|---|---|
| train | 130,319 | 86,821 | 43,498 (33.4%) |
| validation (dev) | 11,873 | 5,928 | 5,945 (50.1%) |

**Key concepts**
- A SQuAD row = `id, title, context, question, answers`. Answers are spans copied from the
  context; `answer_start` is the character offset.
- Unanswerable ⟺ `answers["text"] == []`. (The original JSON uses `is_impossible: true` and
  `plausible_answers`; the HF version drops these and uses empty lists.)
- Dev questions have multiple reference answers (one per annotator). Scoring takes the
  **max** over references, so a correct answer isn't penalized for a different span boundary.
- EM (exact match) is 0 or 1 per question after normalization (lowercase, strip punctuation,
  remove a/an/the, collapse whitespace); reported EM is the average over questions.
- Overall EM hides behavior: an "always abstain" model gets 50.1% overall EM on dev, but
  HasAns EM = 0% and over-abstention = 100% (while hallucination rate = 0% looks perfect).
  Hallucination and over-abstention must be reported together.
- Unanswerable questions are adversarial: they reuse the passage's words and change one
  detail (shifted date, reversed relation, swapped entity). A keyword-overlap rule
  ("abstain if a question word is missing from the passage") abstained on all 9 questions of
  the first dev passage, answerable ones included. Word presence doesn't separate the classes.

**Decisions & why**
- Python 3.11 (not system 3.13) to match the GPU cluster and vLLM/bitsandbytes support.
- Evaluate on `validation` (SQuAD's test set is hidden); `train` is for fine-tuning only.
- Stages 0–1 run on the Mac (M4 Pro, 24 GB): CPU-only work. Stage 2+ code is written on the
  Mac and run on the university NVIDIA GPU, since vLLM and bitsandbytes need CUDA and an 8B
  model needs ~16 GB for weights in 16-bit.
- Note for Stage 6: train is 33% unanswerable but dev is 50%, so there's a class-balance shift.

**Open items**
- GPU model/VRAM, how the cluster is accessed (Slurm/Jupyter/SSH), internet on compute nodes.
- ~~Llama 3 gated-model access on HF~~: approved; Mac logged in as `djdipin` and verified by
  downloading `config.json`. Still to do: `hf auth login` on the cluster.
