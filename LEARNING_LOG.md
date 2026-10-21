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
- ~~GPU model/VRAM, how the cluster is accessed~~: RTX 4090 24 GB on Windows 11, accessed via
  VS Code tunnel. vLLM will run in Docker Desktop (GPU access verified with `nvidia-smi` in a
  CUDA container); everything else runs natively on Windows.
- ~~Llama 3 gated-model access on HF~~: approved; Mac logged in as `djdipin` and verified by
  downloading `config.json`. GPU machine also logged in (`djdipin`) and verified the same way.

## Build sprint: Stages 2–7 (started 2026-10-19)

GPU access ends Wed 2026-10-21 12pm, so we're building first and explaining later
(stage gates paused; Stage 1 check-understanding questions still unanswered).

**What we built**
- vLLM server in Docker (`vllm/vllm-openai:v0.31.0`, model served as `llama3-8b-instruct`,
  bf16, `--gpu-memory-utilization 0.85` → 3.75 GiB KV cache ≈ 30.7k tokens). Model cached
  in the Docker volume `hf-cache`.
- `prompts.py`: 4 prompt variants, rule-based answer parser, judge prompt, self-check prompt.
- `pipeline.py`: LangGraph graph load → prompt → generate → parse → [judge] → score. Saves
  every raw output to `outputs/<run>.jsonl` and metrics to `results/<run>.json`.
- `Dockerfile.train` + `train.py`: QLoRA training in a Linux container (peft 0.21.2,
  trl 1.16.0, bitsandbytes 0.50.2, transformers 5.17.0, torch 2.13.0+cu130).

**Decisions & why**
- Every result is scored two ways: `parse` (regex parser) and `judge` (LLM judge decides
  abstain vs answer; the answer text always comes from the parser).
- Judge v1 extracted the "shortest answer phrase": it shortened correct spans ("later ARPANET
  architecture" → "ARPANET architecture") and marked real answers as ABSTAIN. Dropped.
- Judge v2 saw the question and still tended to judge *correctness*. Judge v3 sees only the
  response: agreed with the parser on 198/200 vs 188/200 (zero_shot, dev n=200 subset).
  Agreement with the parser ≠ correctness; human labels (Stage 5) will measure that.
- ⚠️ The dev n=200 subset (`shuffle(seed=0)`, first 200) was used to develop the judge and
  parser, so it must be **excluded** from the human-label validation sample.
- Self-check v1 ("check your answer" as a follow-up turn) made LLaMA apologize and retract
  correct answers (sycophancy), and ramble so the span was lost: HasAns EM 0.95% on n=200.
  v2 is a separate verifier call (SUPPORTED/UNSUPPORTED); on n=200 it rejected 6 of 140.

**Real numbers: prompt variants, full dev (11,873 questions), parser scoring**
| Variant | EM | F1 | HasAns EM | NoAns acc | Halluc. | Over-abst. | Time |
|---|---|---|---|---|---|---|---|
| zero_shot | 36.71 | 47.31 | 55.35 | 18.12 | 81.88 | 0.12 | 118 s |
| abstain | 54.12 | 64.67 | 52.88 | 55.36 | 44.64 | 1.80 | 112 s |
| few_shot | 64.47 | 70.01 | 77.90 | 51.08 | 48.92 | 1.84 | 109 s |
| self_check | 58.58 | 68.84 | 53.02 | 64.12 | 35.88 | 3.88 | 167 s |
Judge-scored versions are in `results/*__full.json` (within ~2 points; the judge flags more
abstentions, so hallucination is lower and over-abstention higher).

**QLoRA timing run** (all modules, r=64, 30 steps × batch 16): 125 s, ~3.85 examples/s,
peak GPU 18.65 GB, loss 0.73 → 0.25. Trainable params 167,772,160 / 8,198,033,408 (2.05%).
→ Grid uses 10,000 train examples per config, 1 epoch (~45 min/run, ~9 h for 12 runs).
- transformers 5 removed `warmup_ratio`; `warmup_steps=0.03` (fraction) replaces it.
- qv runs peaked at 21.5 GB (of ~22.4 GB free) with batch 16. From qkvo on, batch 8 × grad
  accumulation 2: same effective batch (16) and step count (625), roughly half the activation
  memory. Small confound to mention when comparing qv to qkvo/all.

**Real numbers: QLoRA grid** (10,000 train examples, 1 epoch = 625 steps, effective batch 16,
lr 2e-4, alpha = 2r; all 12 runs finished 2026-10-20 01:41, all exit 0)
| Config | Trainable params | Trainable % | Train loss | Time | Peak GPU | Batch |
|---|---|---|---|---|---|---|
| qv-r8 | 3,407,872 | 0.042 | 0.2426 | 1,770 s | 21.50 GB | 16×1 |
| qv-r16 | 6,815,744 | 0.085 | 0.2385 | 1,767 s | 21.52 GB | 16×1 |
| qv-r32 | 13,631,488 | 0.169 | 0.2362 | 1,768 s | 21.55 GB | 16×1 |
| qv-r64 | 27,262,976 | 0.338 | 0.2387 | 1,768 s | 21.63 GB | 16×1 |
| qkvo-r8 | 6,815,744 | 0.085 | 0.2368 | 1,624 s | 14.62 GB | 8×2 |
| qkvo-r16 | 13,631,488 | 0.169 | 0.2346 | 1,623 s | 14.67 GB | 8×2 |
| qkvo-r32 | 27,262,976 | 0.338 | 0.2368 | 1,624 s | 14.77 GB | 8×2 |
| qkvo-r64 | 54,525,952 | 0.674 | 0.2469 | 1,626 s | 14.98 GB | 8×2 |
| all-r8 | 20,971,520 | 0.260 | 0.2339 | 2,150 s | 15.10 GB | 8×2 |
| all-r16 | 41,943,040 | 0.520 | 0.2374 | 2,150 s | 15.26 GB | 8×2 |
| all-r32 | 83,886,080 | 1.034 | 0.2480 | 2,151 s | 15.57 GB | 8×2 |
| all-r64 | 167,772,160 | 2.046 | 0.2761 | 2,158 s | 16.22 GB | 8×2 |
(Trainable % = trainable / all params as reported by peft; exact values in `adapters/*/train_info.json`.)
- Batch 8×2 dropped peak memory from ~21.5 GB to ~15 GB, and the qkvo runs were *faster*
  than qv at 16×1.
- Train loss gets *worse* at the largest ranks (qkvo-r64, all-r32, all-r64). Likely cause:
  alpha = 2r keeps the LoRA scale fixed, but more trainable params at the same lr 2e-4 means
  bigger effective updates. Loss is on train data; dev metrics decide.
- Sanity check qv-r16 vs base, `abstain` prompt, dev n=200: EM 56.5 → 84.0, hallucination
  40.0 → 15.8%, over-abstention 2.86 → 5.71%.
- Adapters are trained on a 4-bit base but served by vLLM on the bf16 base (standard practice;
  a small train/inference mismatch).

**Real numbers: fine-tuned adapters, full dev (11,873), `abstain` prompt, parser scoring**
| Config | Trainable params | EM | F1 | HasAns EM | NoAns acc | Halluc. | Over-abst. |
|---|---|---|---|---|---|---|---|
| base (no adapter) | 0 | 54.12 | 64.67 | 52.88 | 55.36 | 44.64 | 1.80 |
| qv-r8 | 3,407,872 | 85.34 | 88.38 | 82.89 | 87.79 | 12.21 | 6.16 |
| qv-r16 | 6,815,744 | 85.47 | 88.51 | 83.22 | 87.72 | 12.28 | 5.85 |
| qv-r32 | 13,631,488 | 85.87 | 88.76 | 83.55 | 88.17 | 11.83 | 5.97 |
| qv-r64 | 27,262,976 | 85.91 | 88.85 | 83.08 | 88.73 | 11.27 | 6.33 |
| qkvo-r8 | 6,815,744 | 86.02 | 88.97 | 83.67 | 88.36 | 11.64 | 5.74 |
| qkvo-r16 | 13,631,488 | 85.94 | 88.91 | 83.91 | 87.97 | 12.03 | 5.52 |
| qkvo-r32 | 27,262,976 | 86.07 | 88.96 | 83.49 | 88.65 | 11.35 | 5.97 |
| qkvo-r64 | 54,525,952 | 86.05 | 89.05 | 82.30 | 89.79 | 10.21 | 6.95 |
| all-r8 | 20,971,520 | 86.44 | 89.32 | 83.70 | 89.17 | 10.83 | 6.22 |
| all-r16 | 41,943,040 | **86.50** | **89.43** | 83.45 | 89.54 | 10.46 | 6.34 |
| all-r32 | 83,886,080 | 85.75 | 88.84 | 82.56 | 88.93 | 11.07 | 6.41 |
| all-r64 | 167,772,160 | 85.02 | 88.24 | 80.70 | 89.34 | 10.66 | 7.74 |
Judge-scored versions in `results/<config>__abstain__full.json` (EM ~1.2 lower; judge
over-abstention ~9–11% vs parser ~6%, a disagreement to examine during hand-labelling).
- Fine-tuning is the big effect: +31 EM points and hallucination 44.6% → ~10–12% over the
  same prompt. Every config beats the best prompt-only variant (few_shot, 64.47 EM).
- The whole grid spans only 85.02–86.50 EM. One seed per config, and the binomial standard
  error of EM at n=11,873 is about 0.3 points, so most adjacent differences are within noise.
  all-r64 (most params) is the *worst* EM, consistent with its higher train loss.
- Each adapter eval took ~102–112 s on full dev.

**Real numbers: seed replicates** (`seed_summary.py`; seeds 1–2 trained 2026-10-20 12:06–15:32,
same 10k subset and settings as the grid; full dev, `abstain` prompt, parser scoring)
| Config | EM seed 0 / 1 / 2 | EM mean ± std | Halluc. mean ± std (range) | Over-abst. mean ± std |
|---|---|---|---|---|
| qv-r8 | 85.34 / 85.03 / 85.27 | 85.22 ± 0.16 | 12.70 ± 0.42 (0.76) | 5.84 ± 0.36 |
| all-r16 | 86.50 / 86.26 / 85.77 | 86.18 ± 0.37 | 11.11 ± 0.89 (1.67) | 5.97 ± 0.41 |
| all-r64 | 85.02 / 84.50 / 84.49 | 84.67 ± 0.30 | 12.22 ± 1.46 (2.89) | 6.64 ± 1.07 |
- EM is stable across seeds (std 0.16–0.37). The ordering all-r16 > qv-r8 > all-r64 holds for
  every seed: all three all-r16 seeds beat all three qv-r8 seeds, and all three all-r64 seeds are
  the lowest. With only 3 seeds this is consistent evidence, not a formal significance test.
- Hallucination / over-abstention vary much more across seeds (all-r64 hallucination ranges
  10.66–13.56). Where a run lands on the answer-vs-abstain trade-off is partly luck, so the
  single-seed grid's hallucination differences (10.21–12.28) are mostly within seed noise.
- Seed std of EM (~0.3) is about the size of the binomial standard error estimated earlier, so
  grid differences under ~0.5–1 EM point shouldn't be read as real.
- Debugging note: 4 replicate evals first failed with `APIConnectionError`. `localhost`
  resolved to IPv6 `[::1]`, served by `wslrelay.exe` (not Docker's IPv4 port), which hung under
  load, probably because two eval loops ran at once. Fixed by using `127.0.0.1` in `pipeline.py`.

**Where things are stored**
- Code, results, raw outputs, logs, `train_info.json`: GitHub `DipinJassal/llama3-squad2-qlora`.
- Adapter weights (12 grid + 6 seed replicates, 1.96 GB, too big for GitHub): private Hugging
  Face repo `djdipin/llama3-squad2-qlora-adapters`, one folder per run.

**Explain later (owed to you)**: written up in `STUDY_GUIDE.md` (2026-10-20), still to go
through together. Remaining human task: label the judge-validation sample
(`python judge_validation.py label`, then `score`).
- Stage 1 check-understanding questions.
- How/where/what commands pulled the vLLM image; image vs container; Docker volumes.
- vLLM: server vs offline, continuous batching, KV cache, `--gpu-memory-utilization`.
- LangGraph: state, nodes, conditional edges, why batch-level nodes.
- Parser vs judge design; why the judge doesn't see the question; sycophancy in self-check.
- QLoRA: 4-bit NF4, double quantization, LoRA rank/alpha, target modules, completion-only
  loss, gradient checkpointing, `print_trainable_parameters`.
