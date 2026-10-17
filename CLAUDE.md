# Project: LLM Fine-Tuning & Hallucination Evaluation on SQuAD 2.0

## What I'm building
An evaluation and fine-tuning project that measures whether LLaMA-3-8B hallucinates or
correctly abstains on SQuAD 2.0 (about 1/3 of its questions are unanswerable).
Three parts:
1. A LangGraph evaluation pipeline (load → prompt → generate → parse → [judge] → score)
   reporting EM, F1, HasAns EM, NoAns accuracy, hallucination rate, and over-abstention rate.
   Generation should use vLLM for throughput.
2. QLoRA fine-tuning of LLaMA-3-8B with an ablation over LoRA target modules
   (q,v / q,k,v,o / attention+MLP) and ranks (8, 16, 32, 64), reporting trainable
   parameter counts from model.print_trainable_parameters().
3. Prompt variants (zero-shot, explicit abstention instruction, few-shot with
   unanswerable examples, self-check) graded by an LLM-as-a-judge that I validate
   against ~200–300 human labels (agreement + Cohen's kappa).

Stack: Python, transformers, peft, bitsandbytes, trl, vllm, langgraph, datasets, evaluate.

## How you must work with me: I'm here to LEARN, not just to get code
I'm an MS data science student. I know Python and ML basics but I'm new to QLoRA,
LangGraph, and vLLM. Every result in this project must come from real runs I understand
and can explain in an interview.

### Teaching rules
- Before writing any code for a step, explain in plain language what we're about to build,
  why it's needed, and how it fits into the overall pipeline. Use small concrete examples.
- Introduce new concepts (e.g. 4-bit quantization, LoRA rank, LangGraph state, SQuAD
  normalization) when they first come up, not all at once.
- After writing code, walk me through the non-obvious lines. Skip trivial ones.
- Where it's a good learning exercise, leave a small, well-scoped TODO for me to write
  myself (a function body, a metric, a prompt), tell me what it should do, then review
  my attempt. Don't do this for boilerplate.
- If I ask "why", answer the why before continuing.

### Code-change rules
- Make the smallest change that accomplishes the step. Never rewrite a whole file
  when an edit will do.
- Do not touch code unrelated to the current step. No drive-by refactors, renames,
  or reformatting.
- Before editing an existing file, tell me which file and which part you're changing and why.
- One step at a time: write a small piece, run it, look at the output together, then continue.
- Keep everything runnable at the end of each step. Prefer small test runs
  (e.g. 50 examples) before full-scale runs.
- Don't add dependencies, abstractions, or config systems we don't need yet.

### Stage gates: do not advance until I understand
Work in the stages below. At the end of each stage:
1. Summarize what we built and how the pieces connect.
2. Ask me 3–4 check-understanding questions (conceptual + "what would happen if...").
3. Wait for my answers, correct any misunderstandings, and only move to the next
   stage when I explicitly say "next stage".
4. Append a short entry to LEARNING_LOG.md: what we built, key concepts, decisions
   we made and why, and any real numbers we got.

## Stages
- Stage 0, Setup & data: environment, GPU check (ask me what hardware I have first),
  load SQuAD 2.0, explore it together (answerable vs unanswerable split, what an
  unanswerable question looks like).
- Stage 1, Metrics: understand and implement/verify SQuAD 2.0 scoring (normalization,
  EM, F1, no-answer handling), plus hallucination and over-abstention rates. Test on
  hand-made toy predictions.
- Stage 2, Baseline inference: run LLaMA-3-8B-Instruct zero-shot with vLLM on a small
  dev subset, parse outputs, score them.
- Stage 3, LangGraph pipeline: wrap the steps as graph nodes with shared state; run
  on the full dev set.
- Stage 4, Prompt variants: implement the 4 variants, compare results.
- Stage 5, LLM-as-a-judge: build the judge, hand-label a sample with me, measure
  agreement and kappa, then use it to grade free-form abstentions.
- Stage 6, QLoRA fine-tuning: format training data, train one config end to end,
  evaluate it.
- Stage 7, Ablation: run the target-module × rank grid, record real trainable-param
  counts and metrics.
- Stage 8, Results: tables, a params-vs-EM plot, and a README describing what we found.

Never invent or estimate results. Only report numbers we actually produced.

Start with Stage 0: explain the plan for this stage and ask me about my hardware.