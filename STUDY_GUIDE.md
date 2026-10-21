# Study guide: understanding what we built

The project was built fast while GPU time was limited. This guide is the "explain later" part: each
concept tied to the code and to the real numbers we got, plus the questions an interviewer is
likely to ask. Try answering each question before opening the answer.

Contents: 1. SQuAD 2.0 metrics · 2. Docker & the vLLM image · 3. vLLM · 4. LangGraph ·
5. Prompt variants, parser, judge · 6. QLoRA · 7. The ablation & seeds · 8. Judge validation &
Cohen's kappa · 9. Debugging stories · 10. Interview questions

---

## 1. SQuAD 2.0 metrics (Stage 1)

**Normalization** (`metrics.py: normalize_answer`): lowercase → drop ASCII punctuation → remove
the words a/an/the → collapse whitespace. `"The Normans."` and `"normans"` both become
`"normans"`. Without this, EM would punish formatting, not knowledge.

**EM** is 0/1 per question: does the normalized prediction equal *any* normalized gold answer?
**F1** is token overlap: precision = shared tokens / prediction tokens, recall = shared tokens /
gold tokens, F1 = their harmonic mean; take the max over gold answers.

**No-answer convention**: prediction `""` = abstained; gold `[]` = unanswerable. Abstaining on an
unanswerable question scores EM = F1 = 1; any answer scores 0. Junk gold answers like `"."`
normalize to `""` and are dropped (the official scorer does this; our cross-check found it).

**Why hallucination and over-abstention together**: each can be gamed alone. Never answering →
0% hallucination; always answering → 0% over-abstention. Dev is 50% unanswerable, so "always
abstain" already gets 50.1% EM.

### Stage 1 check-understanding questions

**Q1.** Gold `"1990–2000"` (en dash), prediction `"1990-2000"` (hyphen). What do they normalize to,
and what's the EM?
<details><summary>Answer</summary>

The hyphen is in `string.punctuation`, so the prediction becomes `"19902000"`. The en dash is a
Unicode character, *not* in `string.punctuation`, so the gold stays `"1990–2000"`. They differ →
EM = 0 (F1 = 0 too: one token each, different). Normalization only knows ASCII punctuation; we keep
this behaviour because the official scorer does the same, so our numbers stay comparable.
</details>

**Q2.** EM and F1 for prediction `"Rollo the Viking"` vs gold `["Rollo"]`?
<details><summary>Answer</summary>

Normalized: `"rollo viking"` (the article is removed) vs `"rollo"`. EM = 0. Tokens: prediction
[rollo, viking], gold [rollo]; overlap = 1. Precision = 1/2, recall = 1/1,
F1 = 2·0.5·1 / (0.5 + 1) = 0.667.
</details>

**Q3.** What if the model never abstains, always outputting some span? NoAns accuracy,
hallucination, over-abstention on dev? Why can't overall EM exceed ~50%?
<details><summary>Answer</summary>

NoAns accuracy 0%, hallucination 100%, over-abstention 0%. Every unanswerable question (5,945 of
11,873) scores 0, so even with perfect answers on the 5,928 answerable ones, EM ≤ 5,928/11,873 =
49.9%. Our zero-shot run came close to this regime: 81.88% hallucination, 36.71 EM.
</details>

**Q4.** On an unanswerable question LLaMA says *"The passage does not say who Charles III swore
fealty to."* Fed straight into `compute_metrics`, how is it scored? What does that tell us?
<details><summary>Answer</summary>

It's a non-empty string, so it counts as an *answer* → scored as a hallucination, though the
model actually abstained. Hallucination would be inflated. That's why we needed a parser (regex for
abstention phrases) and a judge. It really happened: on the first 50-question test, the parser
missed "There is no mention of…" phrasing until we added that pattern.
</details>

---

## 2. Docker and how we got vLLM (the "pull" walkthrough)

- **Image vs container**: an *image* is a read-only snapshot of a filesystem (Linux + Python +
  CUDA libraries + vLLM). A *container* is a running instance of an image. One image, many
  containers. `docker pull` downloads an image; `docker run` starts a container from it.
- **Registry and tag**: `vllm/vllm-openai:v0.31.0` = Docker Hub repository `vllm/vllm-openai`,
  tag `v0.31.0`. We listed tags through Docker Hub's API
  (`hub.docker.com/v2/repositories/vllm/vllm-openai/tags`). The newest was `v0.32.0`, released the
  day before; we **pinned** `v0.31.0` (2 weeks old) so the run is reproducible and less likely to
  hit a fresh bug. `latest` changes over time, so results built on it can't be reproduced.
- **Where it lives**: Docker Desktop on Windows runs a hidden Linux VM (WSL2). Images (~8.6 GB for
  vLLM) are stored inside that VM, not in the project folder.
- **Why Docker at all**: vLLM only runs on Linux. The GPU check
  `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` proved containers
  can see the RTX 4090.
- **Flags we used**:
  - `--gpus all`: give the container the GPU.
  - `-p 8000:8000`: map the container's port 8000 to the host so Windows Python can reach vLLM.
  - `-v hf-cache:/root/.cache/huggingface`: a named *volume* that persists the 16 GB model download
    across containers (vLLM and training share it, so we downloaded once).
  - `-v "${PWD}:/work"`: mount the project folder, so the container can read adapters/ and code.
  - `-e HF_TOKEN`: pass the Hugging Face token from the environment without writing it in the command.
  - `--ipc=host`: PyTorch uses shared memory between processes; the default container limit is tiny.
  - `--entrypoint python3` (smoke test): override the image's default command (the vLLM server).
- **Training image** (`Dockerfile.train`): `FROM vllm/vllm-openai:v0.31.0` (reuse its PyTorch +
  CUDA) and `pip install` pinned peft / trl / bitsandbytes.

---

## 3. vLLM

**Server vs offline**: vLLM can be used as a Python library (`LLM.generate`) or as an HTTP server
with an OpenAI-compatible API. We used the server: our Windows code sends chat requests to
`http://127.0.0.1:8000/v1`, and vLLM applies LLaMA-3's chat template.

**Why it's fast: continuous batching.** A naive loop sends one question, waits, sends the next;
the GPU sits mostly idle. vLLM keeps many requests in flight and adds new ones to the running
batch as soon as others finish. We send up to 256 concurrent requests (`MAX_CONCURRENCY`); a full
dev run (11,873 questions) took ~110 s, roughly 100 questions/second.

**KV cache**: during generation, each token's attention keys and values are cached so they aren't
recomputed. For LLaMA-3-8B: 32 layers × 8 KV heads × 128 dims × 2 (K and V) × 2 bytes = 128 KiB per
token. Our 3.75 GiB of KV cache ÷ 128 KiB = ~30,700 tokens, exactly what vLLM logged
("GPU KV cache size: 30,704 tokens"). That caps how many prompts are processed at once.
**PagedAttention** stores the cache in fixed-size blocks (like OS memory pages) so it isn't wasted
on padding. **Prefix caching** reuses the cache for identical prompt starts: the system prompt (and
in few-shot, the demos) is the same for every question, so it's computed once (one adapter eval run
logged an 83.6% prefix-cache hit rate).

**`--gpu-memory-utilization 0.85`**: the fraction of GPU memory vLLM may take: ~16 GB weights
(8B params × 2 bytes in bf16), activations, CUDA graphs, and the rest becomes KV cache. We left
headroom because Windows itself uses ~1.6 GB of the card.

**LoRA serving**: `--enable-lora --lora-modules name=path` registers adapters; requesting
`model="all-r16"` applies that adapter on top of the shared base weights. One server evaluated all
adapters without reloading the 16 GB base.

---

## 4. LangGraph

`pipeline.py` defines a **StateGraph**:

- **State** (`EvalState`, a TypedDict): the shared dictionary every node reads (config, examples,
  records, metrics).
- **Nodes**: functions that take the state and return a *partial update* (e.g. `parse_node`
  returns `{"records": ...}`), which LangGraph merges into the state.
- **Edges**: `load → prompt → generate → parse`, then a **conditional edge**: `route_after_parse`
  returns `"judge"` if `--judge` was given, else `"score"`. That's the `[judge]` in the diagram.
- `graph.ainvoke(initial_state)` runs it; async nodes (`generate`, `judge`) send concurrent
  requests.

**Design choice: batch-level nodes.** Each node processes all 11,873 records at once instead of
running the graph once per question. That keeps vLLM's batching fed and the graph simple. Per-item
graphs make sense when items take different paths (agents, retries); ours don't.

**Why LangGraph here at all?** It makes the pipeline's structure explicit and swappable (add a
node, add a branch) and gives one state object that's easy to save. Honest answer in an interview:
for a linear eval pipeline, plain functions would also work; LangGraph pays off as branching grows.

---

## 5. Prompt variants, parser, judge

| Variant | Idea | EM | Hallucination | Over-abst. |
|---|---|---|---|---|
| zero_shot | "Answer with a span"; no mention of abstaining | 36.71 | 81.88 | 0.12 |
| abstain | + "if not in the passage, reply exactly: unanswerable" | 54.12 | 44.64 | 1.80 |
| few_shot | abstain + 4 demos (2 answerable, 2 unanswerable) from *train* | 64.47 | 48.92 | 1.84 |
| self_check | abstain, then a separate call: SUPPORTED / UNSUPPORTED | 58.58 | 35.88 | 3.88 |

Observations: telling the model it *may* abstain is the biggest single prompt effect. Few-shot
mostly improves answer *format* (HasAns EM 52.88 → 77.90: answers become short spans). Self-check
trades a little over-abstention for less hallucination.

**Parser** (`parse_answer`): regex for abstention phrases ("unanswerable", "not mentioned", "no
mention", "does not say", ...) → `""`; otherwise take the first line and strip "Answer:" and
quotes.

**Judge** (`JUDGE_SYSTEM`): LLaMA-3-8B classifies each response as ABSTAIN or ANSWER. Three
versions:
1. v1 also extracted "the shortest answer phrase", which *shortened correct answers*
   ("later ARPANET architecture" → "ARPANET architecture", which then failed EM). Removed: the
   answer text always comes from the parser.
2. v2 saw the question, and judged *correctness* (labelled plain wrong-looking answers ABSTAIN).
3. v3 sees **only the response**. Agreement with the parser went 188/200 → 198/200.
   Agreement with the parser is not accuracy. That's what human labels measure (§8).

**Self-check and sycophancy**: v1 continued the conversation with "Check your answer…". LLaMA
answered "I apologize for the mistake…" and retracted *correct* answers, and its rambling broke
the parser (HasAns EM 0.95% on 200 questions). Asking "check your answer" implies it was wrong; RLHF
models tend to agree with implied criticism. v2 is a fresh, neutral verifier call.

**Self-preference caveat**: the judge is the same model it grades. That's why the human-label
check exists.

---

## 6. QLoRA

**LoRA**: instead of updating a weight matrix W (d_out × d_in), freeze it and learn a low-rank
update ΔW = B·A with A (r × d_in) and B (d_out × r). Only A and B train. B starts at zero, so
training starts exactly from the original model. Output = W·x + (α/r)·B·A·x.

**Counting trainable parameters by hand** (each adapted matrix adds r·(d_in + d_out)):
- LLaMA-3-8B: hidden 4096, 32 layers; q,o: 4096→4096; k,v: 4096→1024 (grouped-query attention, 8 KV
  heads); gate,up: 4096→14336; down: 14336→4096.
- q,v per layer: r(4096+4096) + r(4096+1024) = 13,312r → × 32 layers = 425,984r → **r=8: 3,407,872** ✓
- q,k,v,o: 26,624r × 32 = 851,968r → **r=8: 6,815,744** ✓
- + MLP (gate, up, down: 18,432r each): 81,920r × 32 = 2,621,440r → **r=8: 20,971,520** ✓
- Doubling r doubles the count; this matches `print_trainable_parameters()` exactly.

**The "Q": 4-bit quantization** (`BitsAndBytesConfig`):
- **NF4** (4-bit NormalFloat): 16 levels placed to suit normally-distributed weights; 8B params ×
  0.5 bytes ≈ 4–5 GB instead of 16 GB.
- **Double quantization**: the per-block scaling constants are themselves quantized (saves ~0.4
  bits/param).
- **Compute dtype bf16**: weights are dequantized on the fly for each matmul; LoRA weights and
  gradients stay in bf16.

**Other training pieces** (`train.py`):
- `prepare_model_for_kbit_training` + **gradient checkpointing**: don't store every activation;
  recompute them in the backward pass. Trades extra compute (typically ~20–30% slower) for a lot of memory.
- **Completion-only loss**: loss on the answer tokens only, not the passage. We're teaching
  *what to answer*, not how to reproduce Wikipedia.
- **α = 2r**: keeps the scale α/r = 2 constant across ranks.
- **Gradient accumulation**: batch 8 × 2 steps = effective batch 16. Same update, half the
  activation memory (peak went 21.5 GB → ~15 GB).
- **Pad token**: LLaMA-3 has none; we used a reserved special token (padding is masked anyway).

**Serving mismatch**: adapters trained against the 4-bit base, served on the bf16 base by vLLM.
Standard practice; the difference is small but worth naming.

---

## 7. The ablation and the seeds

- 12 configs: {q,v · q,k,v,o · attention+MLP} × r ∈ {8, 16, 32, 64}; 10k train examples, 1 epoch.
- **Result**: everything lands at 85.0–86.5 EM (base with the same prompt: 54.12). Capacity is not
  the bottleneck. Even 3.4M params learn "answer with a span, or say unanswerable".
- **Bigger got worse**: all-r64 (168M params) is lowest, with higher train loss (0.276 vs ~0.24).
  Plausible reason: α/r is fixed, but with many more trainable params the same lr 2e-4 makes larger
  effective updates on 625 steps. A hypothesis, not tested here (a learning-rate sweep would).
- **Seeds**: 3 seeds for qv-r8, all-r16, all-r64. EM std 0.16–0.37, and the ranking held for every
  seed. Hallucination varied much more (all-r64: 10.66–13.56), so single-run hallucination
  differences are mostly noise.
- **Rule of thumb from this**: binomial SE of EM ≈ √(0.86·0.14/11,873) ≈ 0.32 points; seed std is
  similar. Differences under ~0.5–1 point need more seeds before anyone believes them.

---

## 8. Judge validation and Cohen's kappa (your remaining task)

Run `python judge_validation.py label` and press `a` (ANSWER), `b` (ABSTAIN) or `u` (unclear) for
250 responses (you can quit with `q` and resume). Then `python judge_validation.py score`.

- **Why blind**: you don't see the judge's or parser's label, or which prompt produced the
  response.
- **Why exclude the 200 dev questions**: we tuned the judge prompt on them; including them would
  flatter the judge.
- **Agreement** = % of items where judge and human match. **Problem**: if 70% of items are
  ANSWER, a "judge" that always says ANSWER gets 70% agreement.
- **Cohen's kappa** corrects for chance: κ = (p_observed − p_expected) / (1 − p_expected), where
  p_expected = Σ over labels of P(human = label)·P(judge = label). κ = 1 perfect, 0 = chance level.
  Rough reading: > 0.8 strong, 0.6–0.8 substantial. (Tested on the textbook example: κ = 0.40.)
- The script also scores the regex parser against your labels, so you can say which is better.

---

## 9. Debugging stories (good interview material)

1. **The `"."` reference answer**: our scorer disagreed with the official one; three dev
   questions have a junk reference that normalizes to empty, which would have rewarded abstaining.
2. **Judge too clever**: shortening answers and judging correctness (§5).
3. **Self-check sycophancy** (§5).
4. **transformers 5 removed `warmup_ratio`**: replaced by fractional `warmup_steps=0.03`.
5. **Memory near the limit**: peak 21.5 GB of ~22.4 GB free; switched to 8×2 accumulation.
6. **Unseeded LoRA init**: `get_peft_model` runs before the Trainer seeds, so seed-0 inits were
   random. Fixed for replicates with `set_seed`.
7. **`localhost` vs `127.0.0.1`**: on Windows, `localhost` resolved to IPv6 `[::1]`, served by
   the WSL relay instead of Docker's port; it hung under load and evals failed with
   `APIConnectionError`. Found by listing port-8000 listeners (`netstat`) and testing both addresses.

---

## 10. Interview questions to practise

<details><summary>Why does an 8B model hallucinate on unanswerable questions?</summary>

Instruction tuning rewards being helpful; the unanswerable questions are adversarial (they reuse
the passage's words and change one detail), and with no permission to abstain the model picks the
most plausible span. Zero-shot: 81.88% hallucination; permitting abstention halves it.
</details>

<details><summary>Why QLoRA rather than full fine-tuning?</summary>

Full fine-tuning of 8B in bf16 needs the weights (16 GB) plus gradients and Adam states (several
× more), far beyond 24 GB. QLoRA keeps the base frozen in 4-bit (~5 GB) and trains 3–168M
parameters; peak memory 15–21.6 GB, ~30 min per run.
</details>

<details><summary>What does rank control, and what did you find?</summary>

The dimension of the low-rank update: capacity and parameter count (linear in r). Here it barely
mattered (EM range 1.5 points across 49× params), and the largest adapter was worst, consistently
over 3 seeds.
</details>

<details><summary>How do you know the differences between configs are real?</summary>

Mostly they aren't: seed std ≈ 0.2–0.4 EM and binomial SE ≈ 0.3. Only the larger, seed-consistent
gaps (all-r16 > qv-r8 > all-r64) are supported, and with 3 seeds that's evidence, not proof.
</details>

<details><summary>What's the trade-off after fine-tuning?</summary>

Hallucination falls from 44.6% to ~10–12%, but over-abstention rises from 1.8% to ~6%: the model
says "unanswerable" more often when an answer exists. Train is 33% unanswerable vs dev 50%, which
also shapes where it lands.
</details>

<details><summary>How would you improve this?</summary>

Learning-rate sweep per adapter size; train on the full 130k set or more epochs; more seeds;
calibrate abstention with a threshold (e.g. on the probability of "unanswerable") and plot the
hallucination/over-abstention trade-off curve; a stronger or different-family judge, validated
with human labels; evaluate on a held-out split instead of reusing dev for development.
</details>
