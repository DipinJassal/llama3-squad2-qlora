"""LangGraph evaluation pipeline: load -> prompt -> generate -> parse -> [judge] -> score.

Generation goes to a vLLM OpenAI-compatible server (see README). Example:
    python pipeline.py --variant zero_shot --n 50
    python pipeline.py --variant abstain --judge            # full dev set
    python pipeline.py --variant abstain --model qv-r16     # a LoRA adapter served by vLLM
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import TypedDict

from datasets import load_dataset
from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from metrics import compute_metrics
from prompts import (
    ABSTAIN_TOKEN, build_messages, judge_messages, parse_answer, parse_judge,
    pick_few_shot_examples, self_check_messages,
)

# 127.0.0.1, not localhost: on Windows "localhost" can resolve to IPv6 [::1], which goes
# through the WSL relay (wslrelay.exe); it hung under load during the seed-replicate evals.
BASE_URL = "http://127.0.0.1:8000/v1"
BASE_MODEL = "llama3-8b-instruct"   # --served-model-name of the vLLM server
MAX_CONCURRENCY = 256               # in-flight requests; vLLM batches them on the GPU


class EvalState(TypedDict, total=False):
    # config (set by the caller)
    variant: str
    model: str
    n: int | None
    use_judge: bool
    run_name: str
    # filled in by the nodes
    examples: list[dict]
    records: list[dict]
    metrics: dict


async def chat(client, sem, model, messages, max_tokens=64):
    async with sem:
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
        )
    return resp.choices[0].message.content or ""


async def run_all(coros, desc):
    return await tqdm_asyncio.gather(*coros, desc=desc)


# ---------------------------------------------------------------- nodes
def load_node(state: EvalState):
    dev = load_dataset("rajpurkar/squad_v2", split="validation")
    if state.get("n"):
        dev = dev.shuffle(seed=0).select(range(state["n"]))  # fixed random subset
    return {"examples": list(dev)}


def prompt_node(state: EvalState):
    demos = None
    if state["variant"] == "few_shot":
        demos = pick_few_shot_examples(load_dataset("rajpurkar/squad_v2", split="train"))
    records = [
        {
            "id": ex["id"],
            "question": ex["question"],
            "context": ex["context"],
            "gold": ex["answers"]["text"],
            "messages": build_messages(state["variant"], ex["context"], ex["question"], demos),
        }
        for ex in state["examples"]
    ]
    return {"records": records}


async def generate_node(state: EvalState):
    client, sem = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY"), asyncio.Semaphore(MAX_CONCURRENCY)
    records, model = state["records"], state["model"]
    raws = await run_all([chat(client, sem, model, r["messages"]) for r in records], "generate")
    for r, raw in zip(records, raws):
        r["raw"] = raw

    if state["variant"] == "self_check":
        # Second call: the same model verifies its answer (skip if it already abstained).
        todo = [r for r in records if parse_answer(r["raw"]) != ""]
        checks = await run_all([
            chat(client, sem, model,
                 self_check_messages(r["context"], r["question"], parse_answer(r["raw"])),
                 max_tokens=5)
            for r in todo
        ], "self-check")
        for r in records:
            r["first_raw"] = r["raw"]
        for r, check in zip(todo, checks):
            r["check_raw"] = check
            if check.strip().upper().startswith("UNSUPPORTED"):
                r["raw"] = ABSTAIN_TOKEN
    return {"records": records}


def parse_node(state: EvalState):
    for r in state["records"]:
        r["pred_parse"] = parse_answer(r["raw"])
    return {"records": state["records"]}


async def judge_node(state: EvalState):
    # The judge is always the base model, even when evaluating a LoRA adapter.
    client, sem = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY"), asyncio.Semaphore(MAX_CONCURRENCY)
    records = state["records"]
    raws = await run_all(
        [chat(client, sem, BASE_MODEL, judge_messages(r["raw"]), max_tokens=5)
         for r in records],
        "judge",
    )
    for r, raw in zip(records, raws):
        label = parse_judge(raw)
        r["judge_raw"], r["judge_label"] = raw, label
        # The judge only decides abstain vs answer; the answer text comes from the parser.
        # (If the parser abstained but the judge says ANSWER, use the cleaned first line.)
        if label == "ABSTAIN":
            r["pred_judge"] = ""
        elif label == "ANSWER":
            r["pred_judge"] = r["pred_parse"] or r["raw"].strip().splitlines()[0]
        else:  # malformed judge reply: fall back to the rule-based parse
            r["pred_judge"] = r["pred_parse"]
    return {"records": records}


def score_node(state: EvalState):
    records = state["records"]
    golds = [r["gold"] for r in records]
    metrics = {"parse": compute_metrics([r["pred_parse"] for r in records], golds)}
    if state.get("use_judge"):
        metrics["judge"] = compute_metrics([r["pred_judge"] for r in records], golds)
        metrics["judge_invalid"] = sum(r["judge_label"] == "INVALID" for r in records)
    return {"metrics": metrics}


def route_after_parse(state: EvalState):
    return "judge" if state.get("use_judge") else "score"


def build_graph():
    g = StateGraph(EvalState)
    g.add_node("load", load_node)
    g.add_node("prompt", prompt_node)
    g.add_node("generate", generate_node)
    g.add_node("parse", parse_node)
    g.add_node("judge", judge_node)
    g.add_node("score", score_node)
    g.set_entry_point("load")
    g.add_edge("load", "prompt")
    g.add_edge("prompt", "generate")
    g.add_edge("generate", "parse")
    g.add_conditional_edges("parse", route_after_parse, {"judge": "judge", "score": "score"})
    g.add_edge("judge", "score")
    g.add_edge("score", END)
    return g.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="zero_shot",
                    choices=["zero_shot", "abstain", "few_shot", "self_check"])
    ap.add_argument("--model", default=BASE_MODEL, help="served model or LoRA adapter name")
    ap.add_argument("--n", type=int, default=None, help="dev subset size (default: full dev)")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    run_name = args.run_name or f"{args.model}__{args.variant}__{args.n or 'full'}"

    start = time.time()
    final = asyncio.run(build_graph().ainvoke({
        "variant": args.variant, "model": args.model, "n": args.n,
        "use_judge": args.judge, "run_name": run_name,
    }))
    elapsed = time.time() - start

    Path("outputs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    with open(f"outputs/{run_name}.jsonl", "w", encoding="utf-8") as f:
        for r in final["records"]:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("messages", "context")},
                               ensure_ascii=False) + "\n")
    summary = {"run_name": run_name, "variant": args.variant, "model": args.model,
               "n": len(final["records"]), "seconds": round(elapsed, 1), **final["metrics"]}
    Path(f"results/{run_name}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{run_name}  ({len(final['records'])} questions, {elapsed:.0f}s)")
    for source in ("parse", "judge"):
        if source in final["metrics"]:
            print(f"  [{source}] " + "  ".join(
                f"{k}={v:.2f}" for k, v in final["metrics"][source].items() if not k.startswith("n_")))


if __name__ == "__main__":
    main()
